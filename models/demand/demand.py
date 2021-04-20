# base
import pandas as pd
import numpy as np
from collections import Counter
import pickle
from datetime import datetime

# models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from gensim.sklearn_api import D2VTransformer
from gensim.sklearn_api import W2VTransformer

# model utils
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


log_outputs = True

def traversal(root_df, train_top=False):

    # perform level 0
    level_0_df = level_n(root_df, class_='top', n=0, train_top=train_top)
    

    # split up into classes
    # level 1
    classes = level_0_df['level_0_label'].drop_duplicates()     # create list of unique classes ex. [0, 1, 2, 3, 4, 5, 6, ...]
    level_1_df = pd.DataFrame()
    frames = []
    for class_ in classes:                                         # [0, 1, 2, 3, 4] - 0 is one class_ in list classes 
        df = level_0_df[level_0_df['level_0_label'] == class_].copy().reset_index(drop=True)    # find all postings in level_0_df that have class_ that class_ -> if on class 0, find all postings with noc label 0
        df = level_n(df, class_, n=1).reset_index(drop=True)                              # train model on these postings and labels 
        frames.append(df)
    level_1_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)        # concatenate previous df and new df 

    # level 2
    classes = level_1_df['level_1_label'].drop_duplicates()
    level_2_df = pd.DataFrame()
    frames = []
    for class_ in classes:
        df = level_1_df[level_1_df['level_1_label'] == class_].copy().reset_index(drop=True)
        df = level_n(df, class_, n=2).reset_index(drop=True)
        frames.append(df)
    level_2_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    # level 3
    classes = level_2_df['level_2_label'].drop_duplicates()
    level_3_df = pd.DataFrame()
    frames = []
    for class_ in classes:
        df = level_2_df[level_2_df['level_2_label'] == class_].copy().reset_index(drop=True)
        df = level_n(df, class_, n=3).reset_index(drop=True)
        frames.append(df)
    level_3_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    return level_3_df # final dataframe after training - no real change in the data values themselves - all labels are unmodified


def level_n(root_df, class_, n, train_top=False):
    # root_df is the cleaned dataframe
    root_df['noc'] = root_df['noc'].map(lambda x: str(x).zfill(4)) # make sure nocs are all 4 digits by filling in 0s.
    
    # current level tells us which digit to look at for prediction
    root_df[f'level_{n}_label'] = root_df['noc'].astype(str).str[0 : n + 1]
    X = root_df.content
    y = root_df[f'level_{n}_label']
    train_model(X, y, model_name=f'class_{class_}_level_{n}')
    return root_df


def train_model(X, y, model_name):

        # global log_outputs, today

        print('Training model...')
        # check if there is only pair of X and y - if there is don't try to test it
        # if there is only one row (one posting) in the dataset, don't use this dataset
        if len(X) < 3 and len(y) < 3:
            return y

        # find unique y's to find number of classes
        class_count = len(set(y))

        if class_count == 1:
            #clf = RandomForestClassifier(n_jobs=-1)
            clf = DummyClassifier(strategy='most_frequent') # we do not have to perform any specific modelling. Just return the same class for all postings.
        elif class_count == 2:
            clf = LogisticRegression(n_jobs=-1, max_iter=500)
        else:
            clf = LinearSVC(loss='hinge', C=0.1, max_iter=2000)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        if len(set(y_train)) < 2:
            return y # not enough data to train. ignore this set

        nb = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer()),
            ('clf', clf)
            ])

        print('Created pipeline')
        nb.fit(X_train, y_train)
        print('Fitted')

        # find accuracy using X_test and y_test
        y_pred = nb.predict(X_train) # train and test on same data
        accuracy_train = accuracy_score(y_train, y_pred)
        y_pred = nb.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred)
 
        print(f'Model: {model_name}, Train Accuracy: {accuracy_train * 100}%, Test Accuracy: {accuracy_test * 100}%')

        filename = f'models/demand/dumps/{model_name}.sav'
        with open(filename, 'wb') as file_obj:
            pickle.dump(nb, file_obj)

def traversal_predict(root_df):

    # perform level 0
    level_0_df = level_n_predict(root_df, class_='top', n=0)

    # split up into classes
    # level 1
    classes = level_0_df['level_0_pred'].drop_duplicates()     # create list of unique classes ex. [0, 1, 2, 3, 4, 5, 6, ...]
    level_1_df = pd.DataFrame()
    frames = []
    for class_ in classes:                                         # [0, 1, 2, 3, 4] - 0 is one class_ in list classes 
        df = level_0_df[level_0_df['level_0_pred'] == class_].copy().reset_index(drop=True)      # find all postings in level_0_df that have class_ that class_ -> if on class 0, find all postings with noc label 0
        df = level_n_predict(df, class_, n=1).reset_index(drop=True)                              # predict on these postings and labels 
        frames.append(df)
    level_1_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    # level 2
    frames = []
    classes = level_1_df['level_1_pred'].drop_duplicates()
    level_2_df = pd.DataFrame()
    for class_ in classes:
        df = level_1_df[level_1_df['level_1_pred'] == class_].copy().reset_index(drop=True)
        df = level_n_predict(df, class_, n=2).reset_index(drop=True)
        frames.append(df)
    level_2_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    
    # level 3
    frames = []
    classes = level_2_df['level_2_pred'].drop_duplicates()
    level_3_df = pd.DataFrame()
    for class_ in classes:
        df = level_2_df[level_2_df['level_2_pred'] == class_].copy().reset_index(drop=True)
        df = level_n_predict(df, class_, n=3).reset_index(drop=True)
        frames.append(df)
    level_3_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    return level_3_df # final dataframe after training - no real change in the data values themselves - all labels are 


def level_n_predict(root_df, class_, n):
    # root_df is the cleaned dataframe
    root_df['noc'] = root_df['noc'].map(lambda x: str(x).zfill(4)) # make sure nocs are all 4 digits by filling in 0s.
    
    # save current level and label for comparison to our predicted values
    root_df[f'level_{n}_label'] = root_df['noc'].astype(str).str[0 : n + 1]
    X = root_df.content
    y_pred = pd.Series(use_model(X, model_name=f'class_{class_}_level_{n}')) # save returned predictions in a column

    root_df.loc[:, f'level_{n}_pred'] = y_pred # create new column in root_df with prediction values

    return root_df


def use_model(X, model_name):

    filename = f'models/demand/dumps/{model_name}.sav'
    # open model created in train_model
    try:
        with open(filename, 'rb') as file_obj:
            # open file and get access to the model
            model = pickle.load(file_obj)
    except FileNotFoundError:
        return None
    # use that model to predict labels and return the output
    try:
        y_pred = model.predict(X)
        return y_pred
    except:
        return None


if __name__ == "__main__":

    train_dataset_src = '../../data/interim/cleaned_postings.csv'
    # train_dataset_src = '../../data/processed/human_labelled_postings.csv'
    df1 = pd.read_csv(train_dataset_src)
    #df1 = df1.sample(30000)
    # df1 = df1.drop("Unnamed: 0", axis=1)
    # df1.dropna(inplace=True)
    #df1['noc'] = df1['noc'].astype(str)
    #df1 = df1[df1.noc.str.isnumeric()]

    predict_dataset_src = '../../data/processed/all_postings_minus_human_labelled.csv'
    # predict_dataset_src = '../../data/processed/human_labelled_postings.csv'
    df2 = pd.read_csv(predict_dataset_src)

    log_outputs = True
    today = datetime.today().strftime('%Y-%m-%d') # get today's date for filename
    if log_outputs:
        with open(f'outputs/{today}.txt', 'a') as log_file:
            log_file.write(f'--\nTraining on: {train_dataset_src}\n')

    #level_3 = traversal(df1, train_top=False)
    level_3_predictions = traversal_predict(df2)