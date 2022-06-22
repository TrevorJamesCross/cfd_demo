"""
College Football Data Analytics: Toolbox
Author: Trevor Cross
Last Updated: 06/22/22

Series of functions used to extract and analyze data from collegefootballdata.com.
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
from operator import itemgetter
import json

# import visualization libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import snowflake connector
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ---------------------------------
# ---Define SF and API Functions---
# ---------------------------------

# define CFD function
## define a function to make requests
def make_request(url, api_key):
    
    # define headers
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer {}".format(api_key)}

    # return API call as df
    return pd.json_normalize(req.get(url, headers=headers).json())

## define a function to build the URL
def build_url(base_url, section, sub_section='', filters=''):
    
    # combine base_url and section_name
    final_url = join(base_url, section)
    
    # combine w/ sub_section if exists
    if sub_section != '':
        final_url = join(final_url, sub_section)
        
    # combine w/ filters if exists
    if filters != '':
        final_url = final_url + filters
        
    # return final URL
    return final_url

## define a function to build filters
def build_filter(filter_names, filter_plugins):
    final_filter = "?"
    
    # build filter
    for filter_num, filter_name in enumerate(filter_names):
        final_filter =  final_filter + filter_name + "=" + str(filter_plugins[filter_num]) + "&"
    
    # return final filter (remove last '&')
    return final_filter[:-1]

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
    return (','.join([tup[0] + " " + tup[1] for tup in col_info])).replace('.','_')

# define SF functions
## define a function to connect to SF
def connect_to_SF(json_creds_path):
    
    # read JSON & connect to SF
    with open(json_creds_path) as file:
        creds = json.load(file)
        
    conn = snowflake.connector.connect(user=creds['user'],
                                           password=creds['password'],
                                           account=creds['account'],
                                           warehouse=creds['warehouse'],
                                           database=creds['database'],
                                           schema=creds['schema'])
    
    # return connector
    return conn

## define function to create table in SF
def create_table(conn, table_name, col_info):
    
    # create table
    conn.cursor().execute(
        """
        CREATE TABLE
        {}({})
        """.format(table_name, col_info)
        )
    
    # print result
    print("\n >>> Table {} created!".format(table_name.upper()))
    
## define a function to append data into table in SF
def append_data(conn, df, table_name):
    
    # capitalize columns
    df.columns = map(lambda name: name.upper().replace('.','_'), df.columns)
    
    # write to table
    success, num_chunks, num_rows, _ = write_pandas(conn, df, table_name.upper())
    
    # print result
    if success:
        print("\n >>> {} rows appended to table {}!".format(num_rows, table_name.upper()))
    else:
        print("\n >>> Something went wrong...")

# -------------------------------------------
# ---Define Elo Rating Algorithm Functions---
# -------------------------------------------

## define function to get initial Elo rating
def get_init_rat(team_name, fbs_team_list):
    
    # if fbs team, return higher init rating
    if team_name in fbs_team_list:
        return 1500
    
    # if not fbs team, return lower init rating
    else:
        return 1200

## define function to calculate margin of victory bonus
def MOV_mult(home_rat, away_rat, margin):
    n = np.sqrt(15)
    return log_n(abs(margin)+1, n=n) * ( n / (abs(home_rat - away_rat)*10**-3 + n) )

## define function to calculate Elo confidence
def calc_conf(rat_a, rat_b, scaler=400):
    return 1 / ( 1 + pow(10, (rat_b-rat_a)/scaler) )

## define function to calculate new Elo rating
def calc_new_rats(home_rat, away_rat, margin, K=25):
    
    # calc home & away confidence
    home_conf = calc_conf(home_rat, away_rat)
    away_conf = calc_conf(away_rat, home_rat)
    
    # determine actualized home confidence
    if margin > 0:
        home_act = 1
    elif margin < 0:
        home_act = 0
    else:
        home_act = 0.5
    
    # calc actualized away confidence
    away_act = 1 - home_act
    
    # calc margin of victory multiplier
    mult = MOV_mult(home_rat, away_rat, margin)
    
    # calc new home & away ratings
    home_rat_new = home_rat + mult*K*(home_act - home_conf)
    away_rat_new = away_rat + mult*K*(away_act - away_conf)
    
    # return new ratings, confidence, and actualized value
    return (round(home_rat_new), home_conf, home_act), (round(away_rat_new), away_conf, away_act)

## define function to plot ratings
def plot_rats(team_rats, team_name):
    
        # extract dates and ratings
        dates_list = list(map(itemgetter(0), team_rats[team_name]))
        rats_list = list(map(itemgetter(1), team_rats[team_name]))
        
        # define graph styling
        plt.style.use('bmh')
        plt.rcParams["figure.figsize"] = [20,10]
        
        # plot ratings against date
        plt.plot(dates_list, rats_list)
        
        # add title
        plt.title(team_name + ' Elo Ratings')
        
        # adjust xticks
        seasons = []
        season_start = []
        for date in dates_list:
            if date[0:4] not in seasons:
                seasons.append(date[0:4])
                season_start.append(date)
                
        plt.xticks(season_start, rotation=45)
    
# ----------------------------
# ---Define Other Functions---
# ----------------------------

## define a function to calculate log base n
def log_n(x, n=10):
    return np.log(x) / np.log(n)

## define a function to plot confusion matrix
def disp_conf_mat(preds, acts):
    conf_mat = confusion_matrix(acts, preds)
    ConfusionMatrixDisplay(conf_mat).plot()
    
## define a function to take the cartesian product of an arbitrary number of lists
def cart_prod(list_of_lists):
    
    # check argument is list of lists
    for lth in list_of_lists:
        if not isinstance(lth, list):
            raise TypeError("\n >>> The argument should be a list containing only lists.")
    
    # define recursive function
    def inner_cart_prod(list_0, list_1):
        inner_result = []
        for lth in list_0:
            for mth in list_1:
                inner_result.append([lth,mth])
        return inner_result
    
    # define funciton to flatten 2d list
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
