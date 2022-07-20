"""
College Football Data Analytics: Lift Data
Author: Trevor Cross
Last Updated: 07/12/22

Extracts available data from collegefootballdata.com and loads it into
snowflake.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import support functions
from os.path import expanduser, join
import sys
import json

# import toolbox functions
repo_dir = join(expanduser('~'),'CFD_demo')
sys.path.insert(1, repo_dir)
from toolbox import *

# ------------------------------
# ---Define Data Dictionaries---
# ------------------------------

# define path to API credentials
json_api_key = join(expanduser('~'), 'secrets/CFD_API_key.json')

# get API key
with open(json_api_key) as file:
    creds = json.load(file)
    
api_key = creds['api_key']

# define path to SF credentials
json_creds_path = join(expanduser('~'), 'secrets/SF_creds.json')

# connect to SF
conn = connect_to_SF(json_creds_path)

# define filters
empty = ['']
years = list(np.arange(2001,2022))
weeks = list(np.arange(1,10))
seasonTypes = ['regular', 'postseason']

teams_fbs_resp = list(conn.cursor().execute("SELECT school FROM teams_fbs"))
teams_fbs = [''.join(school).replace(' ','%20') for school in teams_fbs_resp]
    
# define base URL
base_url = "https://api.collegefootballdata.com"

# define list of sections
sections = ['drives']

# define dictionary of subsections
subsection_dict = {'drives':['']}

# define filters for subsections
filter_dict = {'':['year', 'seasonType']}

# define filter plugins values dictionary
plugin_dict = {'':empty,
               'year':years,
               'week':weeks,
               'team1':teams_fbs,
               'team2':teams_fbs,
               'seasonType':seasonTypes}

# -----------------------
# ---Append Data to SF---
# -----------------------

# iterate through sections
for sec in sections:
    
    # iterate through subsections
    for subsec in subsection_dict[sec]:
            
            # map subsections to their filters
            filter_names = tuple(filter_dict[subsec])
            filter_plugins = cart_prod(list(map(lambda x:plugin_dict[x], filter_dict[subsec])))
            
            # iterate through plugins
            for plugin in filter_plugins:
                
                # build filter
                final_filter = build_filter(filter_names, plugin)
                
                # build URL
                url = build_url(base_url, sec, subsec, final_filter)
                print(f"\n >>> URL: {url}")
                
                try:
                    # send API request
                    df = make_request(url, api_key)
                    
                    # define df not to append
                    bad_df = pd.DataFrame(data=[[0,0,0,[]]], columns=['team1Wins','team2Wins','ties','games'])
                    
                except:
                    
                    # print status
                    print(">>> Filter N/A")
                
                # append data to SF table if not empty
                if len(df) > 0 and not df.equals(bad_df):
                    
                    # create table in SF
                    if subsec != '':
                        table_name = '_'.join([sec,subsec])
                    else:
                        table_name = sec
                    
                    try:
                        create_table(conn, table_name, get_col_info(df))
                    except:
                        pass
                        
                    # append data to SF table
                    append_data(conn, df, table_name)

# close connection to SF        
conn.close()
