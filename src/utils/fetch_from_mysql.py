from sqlalchemy import create_engine
import pandas as pd
import pymysql
import time

# To be filled with auth info
mysql_user = ''
mysql_password = ''
mysql_host = ''
mysql_db = ''

def run(query, output_file):

    db_connection_str = f'mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}'
    db_connection = create_engine(db_connection_str)

    start_time = time.time()
    db_connection_str = f'mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}'
    dob_connection = create_engine(db_connection_str)

    df = pd.read_sql(query, con=db_connection)
    print("SQL query took " + str(time.time() - start_time) + " seconds.")

    df.to_csv(output_file, index=False)

    return df

if __name__ == "__main__":
    query = '''
    SELECT wedata_jobs.job_posts.title, wedata_jobs.job_posts.content, wedata_jobs.jobs_to_nocs.noc, wedata_jobs.jobs_to_tags.tag 
    FROM wedata_jobs.job_posts, wedata_jobs.jobs_to_nocs, wedata_jobs.jobs_to_tags
    LEFT JOIN wedata_jobs.jobs_to_tags ON wedata_jobs.job_posts.hash=wedata_jobs.jobs_to_tags.hash
    WHERE wedata_jobs.job_posts.hash=wedata_jobs.jobs_to_nocs.hash;
    '''

    query = ''' SELECT * from wedata_jobs.job_posts LIMIT 500'''

    query = ''' SELECT * from wedata_jobs.jobs_to_skills '''

    query = ''' SELECT * from wedata_jobs.skills ''' 

    db_connection_str = f'mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(query, con=db_connection)
