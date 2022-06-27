"""
College Football Data Demo: Elo Rating Evaluation
Author: Trevor Cross
Last Updated: 06/27/22

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
from tqdm import tqdm
from datetime import datetime
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
             where season >= 1969 and season < 2000
             order by start_date
             """
                
game_df = pd.read_sql(game_query, conn)

# obtain fbs team list
fbs_query = """
            select school from teams_fbs
            """

fbs_team_list = pd.read_sql(fbs_query, conn)['SCHOOL'].tolist()

# close connection
conn.close()

# --------------------------
# ---Run Naive Simulation---
# --------------------------

# create lists for preds and acts
naive_preds = []
naive_acts = []

# test naive strat
for game_num, game in tqdm(game_df.iterrows(), desc='Running Naive Sim ', unit=' game', total=game_df.shape[0]):
        
    # check if home team is fbs
    if game['HOME_TEAM'] in fbs_team_list and game['AWAY_TEAM'] not in fbs_team_list:
        home_conf = 0.75
        
    # check if away team is fbs
    elif game["AWAY_TEAM"] in fbs_team_list and game['HOME_TEAM'] not in fbs_team_list:
        home_conf = 0.25
        
    # assume home field advantage
    else:
        home_conf = 0.55
    
    # calc margin
    margin = game['HOME_POINTS'] - game['AWAY_POINTS']
    
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
retain_weights = [0.85]
Ks = [30]
scalers = [350]
MOV_base = [np.exp(1)]

# get parameter permutations
perms = cart_prod([retain_weights, Ks, scalers, MOV_base])

# store perm results
results = []
    
# run elo simulation for each permutation of paramters
for perm in perms:
    
    team_rats = run_elo_sim(game_df, fbs_team_list,
                            retain_weight=perm[0], K=perm[1], scaler=perm[2], MOV_base=perm[3])
        
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
        # for some (dumb) reason, the below loop will not remove all 0.5's & related
        # indices the first go around. So it's ran until we get only the binary
        for index, value in enumerate(acts):
            if value == 0.5:
                del preds[index]
                del acts[index]
    
    # remove ties (naive)
    while len(np.unique(naive_acts)) > 2:
        # for some (dumb) reason, the below loop will not remove all 0.5's & related
        # indices the first go around. So it's ran until we get only the binary
        for index, value in enumerate(naive_acts):
            if value == 0.5:
                del naive_preds[index]
                del naive_acts[index]
    
    # split train and test data (Elo)
    train_preds, test_preds = preds[:int(0.7*len(preds))], preds[int(0.7*len(preds)):]
    train_acts, test_acts = acts[:int(0.7*len(acts))], acts[int(0.7*len(acts)):]
    
    # split train and test data (naive)
    naive_train_preds, naive_test_preds = naive_preds[:int(0.7*len(naive_preds))], naive_preds[int(0.7*len(naive_preds)):]
    naive_train_acts, naive_test_acts = naive_acts[:int(0.7*len(naive_acts))], naive_acts[int(0.7*len(naive_acts)):]
    
    # calc log loss and accuracy of train data
    if len(perms) > 0:
        
        print("\n >>> perm: {}".format(perm))
        
        train_loss = log_loss(train_acts, train_preds)
        print("\n >>> Log Loss (Train Data): {}".format(train_loss))
        
        naive_train_loss = log_loss(naive_train_acts, naive_train_preds)
        print(" >>> Naive Log Loss (Train Data): {}".format(naive_train_loss))
        
        train_acc = accuracy_score(train_acts, list(map(round,train_preds)))
        print("\n >>> Accuracy (Train Data): {}".format(train_acc))
        
        naive_train_acc = accuracy_score(naive_train_acts, list(map(round,naive_train_preds)))
        print(" >>> Naive Accuracy (Train Data): {}".format(naive_train_acc))
        
        print("\n")
        
        results.append( (perm, train_loss, train_acc) )
    
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
