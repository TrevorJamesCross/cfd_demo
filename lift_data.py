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
from toolbox import make_request, build_url, build_filter, cart_prod

# ------------------------------
# ---Define Data Dictionaries---
# ------------------------------

# define filters
years = list(np.arange(2019,2022,1))
weeks = list(np.arange(1,21,1))

# define base URL
base_url = "https://api.collegefootballdata.com"

# define list of sections
sections = ['games']

# define dictionary of subsections
subsection_dict = {'games':['media', 'players', 'teams']}

# define filters for subsections
filter_dict = {'media':['years','weeks'],
               'players':None,
               'teams':None}

# define filter plugins values dictionary
plugin_dict = {'years':years,
               'weeks':weeks}

for sec in sections:
    
    if subsection_dict[sec] is not None:
        for subsec in subsection_dict[sec]:
            
            if filter_dict[subsec] is not None:
                plugins = list(map(lambda x:plugin_dict[x], filter_dict[subsec]))
                result = cart_prod(plugins)
                print(result)