"""
College Football Data Demo: Elo Rating Evaluation
Author: Trevor Cross
Last Updated: 07/12/22

Simuates NCAAF games using an Elo rating algorithm and compares the results 
against a naive method. Incorporates parameter tuning.
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
from tqdm import tqdm
from operator import itemgetter
from sklearn.metrics import log_loss, accuracy_score

# import toolbox functions
repo_dir = join(expanduser('~'),'CFD_demo')
sys.path.insert(1, repo_dir)
from toolbox import *

# ---------------------------
# ---Pull Game Information---
# ---------------------------

# define path to SF credentials
json_creds_path = join(expanduser('~'),'secrets/SF_creds.json')

# connect to SF
conn = connect_to_SF(json_creds_path)

# obtain game data
game_query = """
             select start_date, home_team, home_points, away_team, away_points from games
             where season >= 1970 and season < 2022
             and home_points is not null and away_points is not null
             order by start_date
             """
                
game_df = pd.read_sql(game_query, conn)
    
# obtain fbs team list
fbs_query = """
            select school from teams_fbs
            """

fbs_team_list = pd.read_sql(fbs_query, conn)['SCHOOL'].tolist()

# obtain recruitment data
rec_query = """ 
            select year, team, points from recruiting_teams
            """

rec_pts_df = pd.read_sql(rec_query, conn)
rec_pts_dict = dict(zip( (rec_pts_df['TEAM'] + '-' + rec_pts_df['YEAR'].apply(str)), rec_pts_df['POINTS']))

# obtain preseason polling info
poll_query = """
             select season, polls from rankings
             where week = 1
             order by season;
             """

poll_df = pd.read_sql(poll_query, conn)

poll_dict = dict()
for row in poll_df.itertuples():
    poll_list = json.loads(row[2])
    for poll in poll_list:
        info_list = poll['ranks']
        for info in info_list:
            if info['school'] + '-' + str(row[1]) in poll_dict:
                poll_dict[info['school'] + '-' + str(row[1])].append(info['rank'])
            else:
                poll_dict[info['school'] + '-' + str(row[1])] = []
                poll_dict[info['school'] + '-' + str(row[1])].append(info['rank'])
                
# close connection
conn.close()

# --------------------------
# ---Run Naive Simulation---
# --------------------------

# create lists for preds and acts
naive_preds = []
naive_acts = []

# run naive strat
for game in tqdm(game_df.itertuples(), desc='Running Naive Sim ', unit='game', total=game_df.shape[0]):
        
    # check if home team is fbs
    if game[2] in fbs_team_list and game[4] not in fbs_team_list:
        home_conf = 0.75
        
    # check if away team is fbs
    elif game[4] in fbs_team_list and game[2] not in fbs_team_list:
        home_conf = 0.25
        
    # assume home field advantage
    else:
        home_conf = 0.55
    
    # calc margin
    margin = game[3] - game[5]
    
    # determine victor
    if margin > 0:
        home_act = 1
    elif margin < 0:
        home_act = 0
    else:
        home_act = 0.5
    
    # append preds and acts
    naive_preds.append(home_conf)
    naive_acts.append(home_act)

# ------------------------
# ---Run Elo Simulation---
# ------------------------

# define parameter iterations
retain_weights = [0.7]
rec_weights = [0.475] ## makes slight improvement
rank_weights = [3.5] ## makes slight improvement
hf_advs = [60]
Ks = [32.5]
scalers = [350]

# get parameter permutations
perms = cart_prod([retain_weights, rec_weights, rank_weights, hf_advs, Ks, scalers])

# store perm results
results = []
    
# run elo simulation for each permutation of paramters
for perm in perms:
    team_rats = run_elo_sim(game_df, fbs_team_list, rec_pts_dict, poll_dict,
                            retain_weight=perm[0], rec_weight=perm[1], rank_weight=perm[2],
                            hf_adv=perm[3], K=perm[4], scaler=perm[5])
        
    # ---------------------
    # ---Evaluate Models---
    # ---------------------
    
    # get predicted probabilities & actualized values (Elo)
    preds = []
    acts = []
    for key in team_rats:
        preds.extend(list(map(itemgetter(2), team_rats[key])))
        acts.extend(list(map(itemgetter(3), team_rats[key])))
    
    # remove Nonetype objects (Elo)
    preds = [pred for pred in preds if pred != None]
    acts = [act for act in acts if act != None]
    
    # remove ties (Elo)
    while len(np.unique(acts)) > 2:
        for index, value in enumerate(acts):
            if value == 0.5:
                del preds[index]
                del acts[index]
    
    # remove ties (naive)
    while len(np.unique(naive_acts)) > 2:
        for index, value in enumerate(naive_acts):
            if value == 0.5:
                del naive_preds[index]
                del naive_acts[index]
    
    # split tune and test data (Elo)
    tune_preds, test_preds = preds[:int(0.6*len(preds))], preds[int(0.6*len(preds)):]
    tune_acts, test_acts = acts[:int(0.6*len(acts))], acts[int(0.6*len(acts)):]
    
    # split tune and test data (naive)
    naive_tune_preds, naive_test_preds = naive_preds[:int(0.6*len(naive_preds))], naive_preds[int(0.6*len(naive_preds)):]
    naive_tune_acts, naive_test_acts = naive_acts[:int(0.6*len(naive_acts))], naive_acts[int(0.6*len(naive_acts)):]
    
    # calc log loss and accuracy of tune data
    if len(perms) > 0:
        
        print("\n >>> perm: {}".format(perm))
        
        tune_loss = log_loss(tune_acts, tune_preds)
        print("\n >>> Log Loss (Tune Data): {}".format(tune_loss))
        
        naive_tune_loss = log_loss(naive_tune_acts, naive_tune_preds)
        print(" >>> Naive Log Loss (Tune Data): {}".format(naive_tune_loss))
        
        tune_acc = accuracy_score(tune_acts, list(map(round,tune_preds)))
        print("\n >>> Accuracy (Tune Data): {}".format(tune_acc))
        
        naive_tune_acc = accuracy_score(naive_tune_acts, list(map(round,naive_tune_preds)))
        print(" >>> Naive Accuracy (Tune Data): {}".format(naive_tune_acc))
        
        print("\n")
        
        results.append( (perm, tune_loss, tune_acc) )
    
    # calc log loss and acc of test data
    if len(perms) == 1:
        print("\n >>> perm: {}".format(perm))
        
        test_loss = log_loss(test_acts, test_preds)
        print("\n >>> Log Loss (Test Data): {}".format(test_loss))
        
        naive_test_loss = log_loss(naive_test_acts, naive_test_preds)
        print(" >>> Naive Log Loss Test Data): {}".format(naive_test_loss))
        
        test_acc = accuracy_score(test_acts, list(map(round,test_preds)))
        print("\n >>> Accuracy (Test Data): {}".format(test_acc))
        
        naive_test_acc = accuracy_score(naive_test_acts, list(map(round,naive_test_preds)))
        print(" >>> Naive Accuracy (Test Data): {}".format(naive_test_acc))
        
        print("\n")

# sort perm results by log loss
if len(perms) > 0:
    results = sorted(results, key=itemgetter(1), reverse=False)
    print("\n >>> Lowest Loss Perm: {}".format(results[0]))
