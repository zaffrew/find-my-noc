#%% imports
import pandas as pd
from tqdm import tqdm
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import re
from collections import Counter

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
words = set(nltk.corpus.words.words())
stopwords = gensim.parsing.preprocessing.STOPWORDS
stopwords = Counter(stopwords)

# stemming and lemmatizing given text
def lemmatize_stemming(text):
    return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))

# preprocessing and cleaning text 
def preprocess(text):

    result = []
    # remove punctuation, two letter words, one letter words, removes large words which may not be actual words
    # tokenizing each word - creates a list of each word in postings text 
    tokens = gensim.utils.simple_preprocess(text)
    tagged = nltk.pos_tag(tokens) # [('this', 'DT'), ('is', 'VBZ'), ('sentence', 'NN'), ('with', 'IN'), ('many', 'JJ'), ('words', 'NNS')]

    keep_tags = ['N', 'V']

    for tuple_ in tagged:
        token, tag = tuple_[0], tuple_[1][0] # only consider the first letter of the tag NNS becomes N, VB becomes V
        if (len(token) > 3) and (token in words) and (token not in stopwords) and (not token.isnumeric()) and (tag in keep_tags):
            # if the token isn't a stopword - a, this, in, etc. - and if it's a word that has more than 3 characters
            # add it to the final list
            # also ignore words that are not nouns or verbs
            result.append(lemmatize_stemming(token.lower()))

    return ' '.join(result)

def run(input_file, output_file):

    postings = pd.read_csv(input_file)

    hash_ = postings.hash

    postings['content'] = postings['title'] + ' ' + postings['content'] # join title and content together

    postings = postings[['hash', 'content', 'noc']]

    postings = postings.dropna()

    postings.content = postings.content.map(preprocess)

    postings.noc = postings.noc.astype(str) # convert noc labels to string type
    
    postings['noc'] = postings['noc'].apply(lambda x: x.zfill(4))

    # remove those postings that have a noc code that is invalid
    postings = postings[postings.noc.str.isnumeric()]

    # drop rows that have null values for content
    postings = postings[postings.content != '']

    # save interim dataset
    postings.to_csv(output_file, index=False)

    return postings

# test code
if __name__ == '__main__':

    input_file = '../../data/raw/postings.csv'
    postings = pd.read_csv(input_file)
    postings = postings.sample(20)

    postings['content'] = postings['title'] + ' ' + postings['content'] # join title and content together

    del postings['title']

    content = postings.content.fillna('N/A')
    content = content.map(preprocess)

    noc = postings.noc.fillna('N/A')

    cleaned_postings = pd.DataFrame({'content': content, 'noc': noc})

    # lines to test
    cleaned_postings = cleaned_postings[cleaned_postings.title.astype(str) != '[]']
    cleaned_postings = cleaned_postings[cleaned_postings.content.astype(str) != '[]']

    print(cleaned_postings)


