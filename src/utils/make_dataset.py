import pandas as pd

import fetch_from_mysql
import data_cleaning

import time

postings_src = '../../data/raw/postings.csv'
jobs_to_tags_src = '../../data/raw/jobs_to_tags.csv'
cleaned_postings_src = '../../data/interim/cleaned_postings.csv'
noc_labelled_postings_src = '../../data/processed/noc_labelled_postings.csv'
human_labelled_postings_src = '../../data/processed/human_labelled_postings.csv'

query1 = '''
SELECT wedata_jobs.job_posts.hash, wedata_jobs.job_posts.title, wedata_jobs.job_posts.content, wedata_jobs.jobs_to_nocs.noc
FROM wedata_jobs.job_posts, wedata_jobs.jobs_to_nocs
WHERE wedata_jobs.job_posts.hash=wedata_jobs.jobs_to_nocs.hash;
'''

query2 = '''
SELECT wedata_jobs.jobs_to_tags.hash
FROM wedata_jobs.jobs_to_tags;
'''

query3 = '''
SELECT wedata_jobs.job_posts.title, wedata_jobs.job_posts.content, wedata_jobs.jobs_to_nocs.noc
FROM wedata_jobs.job_posts, wedata_jobs.jobs_to_tags, wedata_jobs.jobs_to_nocs
WHERE wedata_jobs.job_posts.hash=wedata_jobs.jobs_to_tags.hash
AND wedata_jobs.job_posts.hash=wedata_jobs.jobs_to_nocs.hash;
'''

def run():

    # fetches all raw data - job posting title, content, NOC tags
    print("...fetching all raw data from mysql")
    fetch_from_mysql.run(query=query1, output_file=postings_src)
    print(f"Raw data ({postings_src}) saved to disk.")

    # list of records (only hashes) that are human - labelled
    print("...finding which records are human-labelled from mysql")
    fetch_from_mysql.run(query=query2, output_file=jobs_to_tags_src)
    print(f"Tags ({jobs_to_tags_src}) saved to disk.")

    # clean raw data
    start_time = time.time()
    print("...cleaning raw data")
    data_cleaning.run(input_file=postings_src, output_file=cleaned_postings_src)
    print(f"Cleaned data ({cleaned_postings_src}) saved to disk.") # save the raw data
    print("Cleaning took " + str(time.time() - start_time) + " seconds.")

    # filter out those records that are present in the human-labelled list
    print("...filtering human labelled NOC data")
    # noc_labelled_postings = pd.read_csv(noc_labelled_postings_src)
    cleaned_postings = pd.read_csv(cleaned_postings_src)
    tags = pd.read_csv(jobs_to_tags_src)

    human_labelled_postings = pd.merge(cleaned_postings,
                        tags,
                        on = 'hash',
                        how = 'inner')
    
    human_labelled_postings.to_csv(human_labelled_postings_src, index=False)
    print(f"Cleaned human-labelled data ({human_labelled_postings_src}) saved to disk.")
    
    print("...saving interim data")
    print("...processing interim data")
    print("...processing external data")
    print("...")

run()
