"""
College Football Data Analytics: Toolbox
Author: Trevor Cross
Last Updated: 05/25/22

Series of functions used to extract data from collegefootballdata.com.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import support libraries
import requests as req
from os.path import join
import itertools
from tqdm import tqdm

# import snowflake connector
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ------------------------------
# ---Define Project Functions---
# ------------------------------

## define a function to make requests
def make_request(url, api_key):
    
    # define headers
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer {}".format(api_key)}

    # return API call
    return req.get(url, headers=headers).json()

## define a function to build the URL
def build_url(base_url, section, sub_section=None, filters=None):
    
    # combine base_url and section_name
    final_url = join(base_url, section)
    
    # combine w/ sub_section if exists
    if sub_section is not None:
        final_url = join(final_url, sub_section)
        
    # combine w/ filters if exists
    if filters is not None:
        final_url = final_url + "?{}".format(filters)
        
    # return final URL
    return final_url

## define a function to build filters
def build_filter(filters_list, plugin_dict):
    final_filt = ""
    
    # build filter
    for filt_name in filters_list:
        for filt_num in plugin_dict[filt_name]:
            final_filt =  filters_list[0] + "=" + str(plugin_dict[filters_list[0]]) + "&"
    
    # return final filter
    return final_filt

## define a function to get column name-dtype string pairs & arrange them for SF
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

# ----------------------------
# ---Define Other Functions---
# ----------------------------

## define a function to take the cartesian product of an arbitrary number of lists
def cart_prod(list_of_lists):
    
    # check argument is list of lists
    for lth in list_of_lists:
        if not isinstance(lth, list):
            raise TypeError("The argument should be a list containing only lists.")
    
    # define recursive function
    def inner_cart_prod(list_0, list_1):
        inner_result = []
        for lth in list_0:
            for mth in list_1:
                inner_result.append([lth,mth])
        return inner_result
    
    #define funciton to flatten 2d list
    def flatten(nested_list):
        flat_list = []
        for el in nested_list:
            if isinstance(el, list):
                flat_list.extend(el)
            else:
                flat_list.append(el)
        return flat_list
    
    # exhaust recursion
    if len(list_of_lists) > 1:
        result = inner_cart_prod(list_of_lists[0], list_of_lists[1])
        for nth_pos in range(len(list_of_lists)-2):
            result = list(map(flatten, inner_cart_prod(result, list_of_lists[nth_pos+2])))

        return [tuple(lth) for lth in result]
    
    else:
        return [tuple([lth]) for lth in list_of_lists[0]]