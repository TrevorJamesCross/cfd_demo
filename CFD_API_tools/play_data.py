"""
College Football Data Analytics: Play Data
Author: Trevor Cross
Last Updated: 06/22/22

Extract data from collegefootballdata.com and load it into Snowflake.

This script is self-contained to extract play-by-play data and create/append
this data to a snowflake table PLAY_DATA.

Note: The play-by-play becomes very large over only a couple seasons. So it
is suggested that the user only loads data which is five seasons worth or less
at a given time.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import support libraries
import sys
import requests as req
from tqdm import tqdm

# import snowflake connector
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# import toolbox functions
repo_dir = '~/CFD_demo/'
sys.path.insert(1, repo_dir)
from toolbox import *

# ---------------------------
# ---Extract Data from CFD---
# ---------------------------

## define function to make requests
def make_request(url, api_key):
    
    # define headers
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer {}".format(api_key)}

    # return API call
    return req.get(url, headers=headers).json()


## define function to build play data URL and make API calls
def get_play_data(year_iter, week_iter, api_key):
    
    # init play data
    play_data = []
    
    # iterate through years and weeks
    for year in tqdm(year_iter, desc='Gathering play data ', unit='year'):
        for week in week_iter:
            
            # build URL
            url = "https://api.collegefootballdata.com/plays?seasonType=regular&year={}&week={}".format(year,week)
            
            # make API call w/ UDF
            play_data.extend( make_request(url, api_key) )
    
    # return play data as pandas.DataFrame
    return pd.DataFrame(play_data)


## run script section
if __name__ == "__main__":
    
    # predefine API key
    api_key = #<key>
    
    # define years & weeks to get data from
    years = np.arange(2001,2016,1)
    weeks = np.arange(1,21,1)
    
    # gather play data w/ UDF
    play_data = get_play_data(years, weeks, api_key)

# -----------------------
# ---Load Data into SF---
# -----------------------

## define function to get column name-dtype string pairs & arrange them for SF
def get_col_info(df):
    
    # define dictionary to change Python dtype to SF dtype
    dtype_dict = {"bool":"boolean",
                  "object":"string",
                  "int64":"integer",
                  "float64":"float"
                  }
    
    # get column info pairs
    col_info = df.dtypes.reset_index().astype(str).replace(dtype_dict).apply(tuple, axis=1).tolist()
    
    # return each pair as a single string separated by a comma
    return ','.join([tup[0] + " " + tup[1] for tup in col_info])
    
## run script section
if __name__ == "__main__":
    
    #connect to SF
    conn = connect_to_SF()
    
    # get column info w/ UDF
    col_info = get_col_info(play_data)
    
    # prompt to create table
    make_table = input("Create or replace table <play_data>? [Y|N]")
    
    if make_table.upper() == "Y" or make_table.upper() == "YES": 
        conn.cursor().execute(
            """
            CREATE OR REPLACE TABLE
            play_data({})
            """.format(col_info)
            )
        print("Table created!")
    
    # prompt to append data to table
    append_data = input("Append {} rows to table <play_data>? [Y|N]".format(len(play_data)))
    
    if append_data.upper() == "Y" or append_data.upper() == "YES":
        play_data.columns = map(lambda name: name.upper(), play_data.columns)
        success, num_chunks, num_rows, output = write_pandas(conn, play_data, 'PLAY_DATA')
        print("Data appended!")
        
    # close connection to SF
    conn.close()
