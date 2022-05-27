"""
College Football Data Analytics: Lift Data
Author: Trevor Cross
Last Updated: 05/25/22

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
api_key = "4vgTNfKfVVN0undVZijW9OZFNhmnucIYzSECsmrDW0RKGrdeA3+NiqbS7kAq3hUI"

# define filters
years = list(np.arange(2000,2022,1))
weeks = list(np.arange(1,21,1))

# define base URL
base_url = "https://api.collegefootballdata.com"

# define list of sections
sections = ['records','calendar']

# define dictionary of subsections
subsection_dict = {'records':[''],
                   'calendar':['']}

# define filters for subsections
filter_dict = {'':['year']}

# define filter plugins values dictionary
plugin_dict = {'year':years,
               'week':weeks}

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
                print(">>> URL: " + url)
                
                try:
                    # send API request
                    df = make_request(url, api_key)
                    
                except:
                    
                    # print status
                    print(">>> Filter N/A")
                
                # append data to SF table if not empty
                if len(df) > 0:
                    
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
