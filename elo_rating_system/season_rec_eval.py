"""
College Football Data Demo: Season Record Evaluation
Author: Trevor Cross
Last Updated: 07/13/22

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
import json
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
game_query = f"""
             select start_date, home_team, home_points, away_team, away_points, season, season_type from games
             where season = {season}
             and iff(season < 2022, home_points is not null and away_points is not null, true)
             order by start_date
             """
                
game_df = pd.read_sql(game_query, conn)

# obtain fbs team list
fbs_query = """
            select school from teams_fbs
            """

fbs_team_list = pd.read_sql(fbs_query, conn)['SCHOOL'].tolist()

# obtain recruitment data
rec_query =  f""" 
             select year, team, points from recruiting_teams
             where year = {season}
             """

rec_pts_df = pd.read_sql(rec_query, conn)
rec_pts_dict = dict(zip( (rec_pts_df['TEAM'] + '-' + rec_pts_df['YEAR'].apply(str)), rec_pts_df['POINTS']))

# obtain preseason polling info
poll_query = """
             select season, polls from rankings
             where week = 1 and season = {}
             order by season;
             """.format(season)

poll_df = pd.read_sql(poll_query, conn)

poll_dict = dict()
for row in poll_df.itertuples():
    poll_list = json.loads(row[2])
    for poll in poll_list:
        info_list = poll['ranks']
        for info in info_list:
            if f"{info['school']}-{str(row[1])}" in poll_dict:
                poll_dict[f"{info['school']}-{str(row[1])}"].append(info['rank'])
            else:
                poll_dict[f"{info['school']}-{str(row[1])}"] = []
                poll_dict[f"{info['school']}-{str(row[1])}"].append(info['rank'])

# close SF connection
conn.close()

# define path to Elo rating history JSON file
file_path = join(expanduser('~'), 'CFD_demo/elo_rating_system/team_rating_hist.json')

# load JSON file as Python dictionary
team_rats = json_to_dict(file_path)

# ----------------------------
# ---Run Season Simulations---
# ----------------------------

# define parameter iterations
retain_weight = 0.65
rec_weight = 0.65 
rank_weight = 3.75 
hf_adv = 65
ps_mult = 1.20
K = 40
scaler = 375

# run season simulations
num_sims = 2000
func_inputs = [(season, game_df, team_rats, fbs_team_list, rec_pts_dict, poll_dict,
                retain_weight, rec_weight, rank_weight, hf_adv, K, scaler)]*num_sims

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
        date = like_games[0][1]
        mean_rats = np.mean(list(map(itemgetter(2), like_games)))
        mean_conf = np.mean(list(map(itemgetter(3), like_games)))
        mean_act = np.mean(list(map(itemgetter(4), like_games)))
        
        agg_dict[key].append( (date, round(mean_rats), mean_conf, round(mean_act)) )
    
# --------------------------
# ---Evaluate Simulations---
# --------------------------

if season < 2022:
    
    # get true game outcomes
    true_rec_dict = dict()
    for key_num, key in enumerate(team_rats):
        if key in agg_dict:
            true_rec_dict[key] = [(items[1], items[4]) for items in team_rats[key]
                                  if items[1] in list(map(itemgetter(0), agg_dict[key]))]
    
    # get predicted & true actualized values
    pred_acts = []
    true_acts = []
    for key_num, key in enumerate(agg_dict):
        pred_acts.extend(list(map(itemgetter(2), agg_dict[key])))
        true_acts.extend(list(map(itemgetter(1), true_rec_dict[key])))
    
    # remove Nonetype from true_acts
    ## some teams had duplicate/first game in specified season
    true_acts = [act for act in true_acts if act != None]
    
    # print season & number of simulations aggregated
    print(f"\n >>> {season}, Number of Sims: {num_sims}")
    
    # calc log loss & accuracy
    loss = log_loss(true_acts, pred_acts)
    print(f"\n >>> Log Loss: {loss}")
            
    acc = accuracy_score(true_acts, list(map(round, pred_acts)))
    print(f"\n >>> Accuracy: {acc}")
    
    # evaluate a team's season record
    team_name = 'Wisconsin'
    team_rec, perc_corr = eval_rec(agg_dict, true_rec_dict, team_name)
    
    print(f"\n {team_name}'s Record Prediction: {team_rec} {perc_corr}")