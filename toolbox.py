"""
College Football Data Demo: Toolbox
Author: Trevor Cross
Last Updated: 07/12/22

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
import json
from os.path import join
from tqdm import tqdm
from operator import itemgetter
from datetime import datetime

# import visualization libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

# import snowflake connector
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ---------------------------------
# ---Define SF and API Functions---
# ---------------------------------

# define CFD function
## define function to make requests
def make_request(url, api_key):
    
    # define headers
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer {}".format(api_key)}

    # return API call as df
    return pd.DataFrame(req.get(url, headers=headers).json())

## define function to build the URL
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

## define function to build filters
def build_filter(filter_names, filter_plugins):
    final_filter = "?"
    
    # build filter
    for filter_num, filter_name in enumerate(filter_names):
        final_filter =  final_filter + filter_name + "=" + str(filter_plugins[filter_num]) + "&"
    
    # return final filter (remove last '&')
    return final_filter[:-1]

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
    return (','.join([tup[0] + " " + tup[1] for tup in col_info])).replace('.','_')

# define SF functions
## define function to connect to SF
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
    
## define function to append data into table in SF
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
    return np.log(abs(margin)+1) * ( 2.2 / (abs(home_rat - away_rat)*0.001 + 2.2) )

## define function to calculate Elo confidence
def calc_conf(rat_a, rat_b, gt_mult=1.0, scaler=400):
    return 1 / ( 1 + pow(10, gt_mult*(rat_b-rat_a)/scaler) )

## define function to calculate new Elo rating
def calc_new_rats(home_rat, away_rat, margin, hf_adv=0, gt_mult=1.0,
                  K=25, scaler=400):

    # calc home & away confidence
    home_conf = calc_conf(home_rat+hf_adv, away_rat, gt_mult=gt_mult, scaler=scaler)
    away_conf = 1 - home_conf
    
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
    mov_mult = MOV_mult(home_rat, away_rat, margin)
    
    # calc new home & away ratings
    home_rat_new = home_rat + mov_mult*K*(home_act - home_conf)
    away_rat_new = away_rat + mov_mult*K*(away_act - away_conf)
    
    # return new ratings, confidence, and actualized value
    return (round(home_rat_new), home_conf, home_act), (round(away_rat_new), away_conf, away_act)

## define function to record matchup history in dictionary
def build_matchup_dict(game_df, current_season):
    
    # define matchup dictionary
    matchup_dict = dict()
    
    # iterate through games
    for game in game_df.itertuples():
        
        # check if matchup is in dict
        if game[2] + '-' + game[4] + '-' + str(current_season)  in matchup_dict or game[4] + '-' + game[2] + '-' + str(current_season) in matchup_dict:
            try:
                matchup_dict[game[2] + '-' + game[4] + '-' + str(current_season)].append( game[3]>game[5] )
            except KeyError:
                matchup_dict[game[4] + '-' + game[2] + '-' + str(current_season)].append( game[5]>game[3] )
                
        # append matchup to dict
        else:
            matchup_dict[game[2] + '-' + game[4] + '-' + str(current_season)] = []
            matchup_dict[game[2] + '-' + game[4] + '-' + str(current_season)].append( game[3]>game[5] )
    
    # aggregate results
    for key in matchup_dict:
        matchup_dict[key] = (matchup_dict[key].count(True), matchup_dict[key].count(False))
    
    # return matchup_dict
    return matchup_dict
    
## define function to run elo simulation
def run_elo_sim(game_df, fbs_team_list, rec_pts_dict, poll_dict,
                retain_weight=0.90, rec_weight=0.1, rank_weight=5, hf_adv=0, ps_mult=1.2,
                K=25, scaler=400):
    
    # create dictionary to record team Elo ratings
    team_rats = dict()
    
    # create last_season variable
    last_season = 0
    
    # iterate through games
    for game in tqdm(game_df.itertuples(), desc='Running Elo Sim ', unit=' games', total=game_df.shape[0]):

        # parse current date
        date = str(datetime.strptime(game[1][:10], '%Y-%m-%d').date())
        
        # if home team exists and in same season
        if game[2] in team_rats and game[6] == team_rats[game[2]][-1][0]:
            
            # get current home rating
            home_rat = team_rats[game[2]][-1][2]
        
        # if home team exists and NOT in same season
        elif game[2] in team_rats:
            
            # get initial rating
            init_rat = get_init_rat(game[2], fbs_team_list)
            
            # calc recruitment bonus
            if game[6] > 2003 and f'{game[2]}-{game[6]}' in rec_pts_dict:
                rec_bonus = rec_weight * float(rec_pts_dict[f'{game[2]}-{game[6]}'])
            else:
                rec_bonus = 0
            
            # calc rank bonus
            if f'{game[2]}-{game[6]}' in poll_dict:
                rank_bonus = (26 - np.mean(poll_dict[f'{game[2]}-{game[6]}'])) * rank_weight
            else:
                rank_bonus = 0
            
            # reset home rating
            home_rat = int(retain_weight*(team_rats[game[2]][-1][2] - init_rat + rec_bonus) + init_rat + rank_bonus)
        
        # if NOT home team exists
        else:
            
            # get initial rating
            init_rat = get_init_rat(game[2], fbs_team_list)
            
            # append home team to dict
            team_rats[game[2]] = [(game[6], date, init_rat, None, None)]
            home_rat = team_rats[game[2]][-1][2]
        
        # if away team exists and in same season
        if game[4] in team_rats and game[6] == team_rats[game[4]][-1][0]:
            
            # get current home rating
            away_rat = team_rats[game[4]][-1][2]
        
        # if away team exists and NOT in same season
        elif game[4] in team_rats:
            
            # get initial rating
            init_rat = get_init_rat(game[4], fbs_team_list)
            
            # calc recruitment bonus
            if game[6] > 2003 and f'{game[4]}-{game[6]}' in rec_pts_dict:
                rec_bonus = rec_weight * float(rec_pts_dict[f'{game[4]}-{game[6]}'])
            else:
                rec_bonus = 0
            
            # calc rank bonus
            if f'{game[4]}-{game[6]}' in poll_dict:
                rank_bonus = (26 - np.mean(poll_dict[f'{game[4]}-{game[6]}'])) * rank_weight
            else:
                rank_bonus = 0
            
            # reset away rating
            away_rat = int(retain_weight*(team_rats[game[4]][-1][2]- init_rat + rec_bonus) + init_rat + rank_bonus)
        
        # if NOT away team exists
        else:
            
            # get initial rating
            init_rat = get_init_rat(game[4], fbs_team_list)
            
            # append away team to dict
            team_rats[game[4]] = [(game[6], date, init_rat, None, None)]
            away_rat = team_rats[game[4]][-1][2]
        
        # redefine last_season
        last_season = game[6]
        
        # calc score margin from game
        margin = game[3] - game[5]
        
        # determine game type multiplier
        if game[7] == 'postseason':
            gt_mult = ps_mult
        else:
            gt_mult = 1.0
        
        # calc new ratings
        home_info, away_info = calc_new_rats(home_rat, away_rat, margin, hf_adv=hf_adv, gt_mult=gt_mult,
                                             K=K, scaler=scaler)
        
        home_rat_new, home_conf, home_act = home_info
        away_rat_new, away_conf, away_act = away_info
        
        # append new ratings to dict
        team_rats[game[2]].append( (game[6], date, home_rat_new, home_conf, home_act) )
        team_rats[game[4]].append( (game[6], date, away_rat_new, away_conf, away_act) )

    # return dictionary of team Elo ratings
    return team_rats
    
## define function to plot ratings
def plot_rats(team_rats, team_name, previous_seasons=10):
        
        # create figure
        plt.figure()
        
        # extract dates and ratings
        dates_list = list(map(itemgetter(0), team_rats[team_name]))
        rats_list = list(map(itemgetter(1), team_rats[team_name]))
        
        # filter dates and ratings
        while int(dates_list[0][:4]) < int(dates_list[-1][:4]) - previous_seasons:
            for date_index, date in enumerate(dates_list):
                if int(date[:4]) < int(dates_list[-1][:4]) - previous_seasons:
                    del dates_list[date_index]
                    del rats_list[date_index]
        
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
            if date[:4] not in seasons:
                seasons.append(date[:4])
                season_start.append(date)
                
        plt.xticks(season_start, rotation=45)

# ----------------------------------------
# ---Define Record Prediction Functions---
# ----------------------------------------

## define function to calc predicted margin of victory
def calc_pred_MOV(home_rat, away_rat, pts_per_dif=25):
    return abs(away_rat-home_rat) // pts_per_dif

## define function to sample game results
def sample_game_results(home_rat, away_rat, hf_adv=0, gt_mult=1.0,
                        K=25, scaler=400):
    
    # calc confidence
    home_conf = calc_conf(home_rat+hf_adv, away_rat, gt_mult, scaler)
    away_conf = 1 - home_conf
    
    # generate random number
    rand_num = np.random.uniform(low=0, high=1)
    
    # determine victor
    if home_conf > rand_num:
        home_act = 1
    else:
        home_act = 0
    
    # calc away actualized value
    away_act = 1 - home_act
    
    # calc predicted MOV
    pred_margin = calc_pred_MOV(home_rat, away_rat)
    
    # calc MOV multiplier
    mult = MOV_mult(home_rat, away_rat, pred_margin)
    
    # calc new home & away ratings
    home_rat_new = home_rat + mult*K*(home_act - home_conf)
    away_rat_new = away_rat + mult*K*(away_act - away_conf)
    
    # return new ratings, confidence, and actualized value
    return (round(home_rat_new), home_conf, home_act), (round(away_rat_new), away_conf, away_act)
    

## define function to run record prediction for a season
def run_season_sim(season, game_df, team_rats, fbs_team_list, rec_pts_dict, poll_dict, 
                   retain_weight=0.90, rec_weight=0.1, rank_weight=5, hf_adv=0, ps_mult=1.2,
                   K=25, scaler=400):

    # create dictionary to record hot team ratings
    sim_dict = dict()
    
    # iterate games
    for game in game_df.itertuples():

        # parse current date
        date = str(datetime.strptime(game[1][0:10], '%Y-%m-%d').date())
        
        # if home team already in sim dict
        if game[2] in sim_dict:
            
            # get current home rating
            home_rat = sim_dict[game[2]][-1][2]
        
        # if home team in ratings dict
        elif game[2] in team_rats:
            
            # get initial rating
            init_rat = get_init_rat(game[2], fbs_team_list)
            
            # calc recruitment bonus
            if game[6] > 2003 and f'{game[2]}-{game[6]}' in rec_pts_dict:
                rec_bonus = rec_weight * float(rec_pts_dict[f'{game[2]}-{game[6]}'])
            else:
                rec_bonus = 0
            
            # calc rank bonus
            if f'{game[2]}-{game[6]}' in poll_dict:
                rank_bonus = (26 - np.mean(poll_dict[f'{game[2]}-{game[6]}'])) * rank_weight
            else:
                rank_bonus = 0
            
            # get starting rating for the season
            try:
                home_rat = next(team_rats[game[2]][items_num-1][2] for items_num, items in enumerate(team_rats[game[2]]) if items[0]==game[6])
            except StopIteration:
                home_rat = retain_weight*(team_rats[game[2]][-1][2] - init_rat + rec_bonus) + init_rat + rank_bonus
                

            # append home team to sim dict
            sim_dict[game[2]] = []
        
        # skip game
        else:
            continue
        
        # if away team already in sim dict
        if game[4] in sim_dict:
            
            # get current home rating
            away_rat = sim_dict[game[4]][-1][2]
        
        # if away team in ratings dict
        elif game[4] in team_rats:
            
            # get initial rating
            init_rat = get_init_rat(game[4], fbs_team_list)
            
            # calc recruitment bonus
            if game[6] > 2003 and f'{game[4]}-{game[6]}' in rec_pts_dict:
                rec_bonus = rec_weight * float(rec_pts_dict[f'{game[4]}-{game[6]}'])
            else:
                rec_bonus = 0
            
            # calc rank bonus
            if f'{game[4]}-{game[6]}' in poll_dict:
                rank_bonus = (26 - np.mean(poll_dict[f'{game[4]}-{game[6]}'])) * rank_weight
            else:
                rank_bonus = 0
            
            # get starting rating for the season
            try:
                away_rat = next(team_rats[game[4]][items_num-1][2] for items_num, items in enumerate(team_rats[game[4]]) if items[0]==game[6])
            except StopIteration:
                away_rat = retain_weight*(team_rats[game[4]][-1][2] - init_rat + rec_bonus) + init_rat + rank_bonus
                
            # append away team to sim dict
            sim_dict[game[4]] = []
        
        # skip game
        else:
            
            # remove empty list of home team
            if len(sim_dict[game[2]]) == 0:
                sim_dict.pop(game[2])
                
            continue
        
        # determine game type multiplier
        if game[7] == 'postseason':
            gt_mult = ps_mult
        else:
            gt_mult = 1.0
        
        # sample game results
        home_info, away_info = sample_game_results(home_rat, away_rat, hf_adv=hf_adv, gt_mult=gt_mult,
                                                   K=K, scaler=scaler)
        
        home_rat_new, home_conf, home_act = home_info
        away_rat_new, away_conf, away_act = away_info
        
        # append new ratings to dict
        sim_dict[game[2]].append( (game[6], date, home_rat_new, home_conf, home_act) )
        sim_dict[game[4]].append( (game[6], date, away_rat_new, away_conf, away_act) )
        
    # return dictionary of team Elo ratings
    return sim_dict

## define function to evaluate predicted season record for a given team
def eval_rec(agg_dict, true_rec_dict, team_name):
    game_preds = np.array(list(map(round, map(itemgetter(2), agg_dict[team_name]))))
    true_acts = np.array(list(map(itemgetter(1), true_rec_dict[team_name])))
    
    return game_preds == true_acts, sum(game_preds == true_acts) / len(game_preds)

# ----------------------------
# ---Define Other Functions---
# ----------------------------

## define function to round numbers up
def round_up(x):
    return int(x) + (x % 1 > 0)

## define function to calculate log base n
def log_n(x, n=10):
    return np.log(x) / np.log(n)

## define function to plot confusion matrix
def disp_conf_mat(preds, acts):
    conf_mat = confusion_matrix(acts, preds)
    ConfusionMatrixDisplay(conf_mat).plot()
    
## define function to take the cartesian product of an arbitrary number of lists
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

## define function to save dictionary as JSON file locally
def dict_to_json(my_dict, file_path):
    with open(file_path, "w+") as file:
        json.dump(my_dict, file)

## define function to load local JSON file as Python dict
def json_to_dict(file_path):
    with open(file_path) as file:
        return json.load(file)
    
## define function to plot calibration curve
def plot_cal_curve(acts, preds, n_bins=10):
    probs_act, probs_pred = calibration_curve(acts, preds, n_bins=n_bins)
    
    plt.plot([0,1],[0,1],':k')
    plt.plot(probs_pred, probs_act, '-s')
    
    plt.title('Calibration Curve')
    plt.xlabel("Mean Predicted Value")
    plt.ylabel("Fraction of Positives")