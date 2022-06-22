"""
College Football Data Analytics: Lift Data
Author: Trevor Cross
Last Updated: 06/22/22

Extracts available data from collegefootballdata.com and loads it into
snowflake.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import toolbox functions
from toolbox import *

# ------------------------------
# ---Define Data Dictionaries---
# ------------------------------

# connect to SF
conn = connect_to_SF()

# predefine API key
api_key = #<key>

# define filters
empty = ['']
years = list(np.arange(1869,2000,1))
weeks = list(np.arange(1,21,1))

teams_fbs_resp = list(conn.cursor().execute("SELECT school FROM teams_fbs"))
teams_fbs = [''.join(school).replace(' ','%20') for school in teams_fbs_resp]
    
# define base URL
base_url = "https://api.collegefootballdata.com"

# define list of sections
sections = ['games']

# define dictionary of subsections
subsection_dict = {'games':['']}

# define filters for subsections
filter_dict = {'':['year','week']}

# define filter plugins values dictionary
plugin_dict = {'':empty,
               'year':years,
               'week':weeks,
               'team1':teams_fbs,
               'team2':teams_fbs}

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
                print("\n >>> URL: " + url)
                
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
                        print()
                        
                    # append data to SF table
                    append_data(conn, df, table_name)

# close connection to SF        
conn.close()
