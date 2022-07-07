"""
College Football Data Demo: Season Record Evaluation
Author: Trevor Cross
Last Updated: 07/07/22

Simuates NCAAF games using an Elo rating algorithm to predict end season
records.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import parallelization libraries
from multiprocessing import Pool

# import support functions
from os.path import expanduser, join
import sys
from operator import itemgetter
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score

# import toolbox functions
repo_dir = join(expanduser('~'), 'CFD_demo')
sys.path.insert(1, repo_dir)
from toolbox import *

# ---------------
# ---Pull Data---
# ---------------

# define season and games to simulate
season = 2022

# define path to SF credentials
json_creds_path = join(expanduser('~'),'secrets/SF_creds.json')

# connect to SF
conn = connect_to_SF(json_creds_path)

# obtain game data
game_query = """
             select start_date, home_team, home_points, away_team, away_points from games
             where season = {} and (home_team in (select home_team from games where season = {}))
             order by start_date
             """.format(season, season-1)
                
game_df = pd.read_sql(game_query, conn)

# obtain fbs team list
fbs_query = """
            select school from teams_fbs
            """

fbs_team_list = pd.read_sql(fbs_query, conn)['SCHOOL'].tolist()

# close SF connection
conn.close()

# define path to Elo rating history JSON file
file_path = join(expanduser('~'), 'CFD_demo/elo_rating_system/team_rating_hist.json')

# load JSON file as Python dictionary
team_rats = json_to_dict(file_path)

# ----------------------------
# ---Run Season Simulations---
# ----------------------------

# define Elo rating algorithm parameters
K = 25
scaler = 300
margin = 7

# run season simulations
num_sims = 500
func_inputs = [(season, game_df, team_rats, margin, K, scaler)]*num_sims

with Pool() as pool:
    list_of_sims = pool.starmap(run_season_sim, tqdm(func_inputs, desc="Running Sims ", unit=' sims'))

# "invert" list_of_sims (list of dicts -> dict of lists)
team_sims_dict = dict()
for key_num, key in enumerate(list_of_sims[0]):
    team_sims_dict[key] = []
    
    for sim_num, sim in enumerate(list_of_sims):
        team_sims_dict[key].append(sim[key])

# create aggregate dictionary
agg_dict = dict()
for key_num, key in enumerate(team_sims_dict):
    agg_dict[key] = []
    
    for items_num in range(len(team_sims_dict[key][0])):
        like_games = list(map(itemgetter(items_num), team_sims_dict[key]))
        date = like_games[0][0]
        mean_rats = np.mean(list(map(itemgetter(1), like_games)))
        mean_conf = np.mean(list(map(itemgetter(2), like_games)))
        mean_act = np.mean(list(map(itemgetter(3), like_games)))
        
        agg_dict[key].append( (date, mean_rats, mean_conf, mean_act) )
    
# --------------------------
# ---Evaluate Simulations---
# --------------------------

if season < 2022:
    
    # get true game outcomes
    true_rec_dict = dict()
    for key_num, key in enumerate(team_rats):
        true_rec_dict[key] = [(items[0], items[3]) for items in team_rats[key] if items[0][:4] == str(season)]
    
    # get predicted & true actualized values
    pred_acts = []
    true_acts = []
    for key_num, key in enumerate(agg_dict):
        pred_acts.extend(list(map(itemgetter(3), agg_dict[key])))
        true_acts.extend(list(map(itemgetter(1), true_rec_dict[key])))
    
    # remove Nonetype from true_acts
    ## some teams had duplicate/first game in specified season
    true_acts = [act for act in true_acts if act != None]
    
    # print season & number of simulations aggregated
    print("\n >>> {}, Number of Sims: {}".format(season, num_sims))
    
    # calc log loss & accuracy
    loss = log_loss(true_acts, pred_acts)
    print("\n >>> Log Loss: {}".format(loss))
            
    acc = accuracy_score(true_acts, list(map(round, pred_acts)))
    print("\n >>> Accuracy: {}".format(acc))
    
    # evaluate Wisconsin's season record
    team_name = 'Illinois'
    wisco_rec, perc_corr = eval_rec(agg_dict, true_rec_dict, team_name)
    
    print("\n {}'s Record Prediction: {} {}".format(team_name, wisco_rec, perc_corr))