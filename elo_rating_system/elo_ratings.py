"""
College Football Data Analytics: Elo Ratings
Author: Trevor Cross
Last Updated: 06/23/22

Simuates NCAAF games using an Elo rating algorithm.
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
             where season >= 1969 and season < 1990
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
retain_weights = [0.90]
Ks = [50]
scalers = [450]

# get parameter permutations
perms = cart_prod([retain_weights, Ks, scalers])

results_dict = dict()

# run elo simulation for each permutation of paramters
for perm in perms:
    
    # define reset weight
    retain_weight = perm[0] ## retain_weight=1 => no reset, retain_weight=0 => reset to init_rat
    
    # create dictionary to hold team Elo ratings
    team_rats = dict()
    
    # iterate through games
    for game_num, game in tqdm(game_df.iterrows(), desc='Running Elo Sim ', unit=' game', total=game_df.shape[0]):
        
        # parse current date
        date = str(datetime.strptime(game['START_DATE'][0:10], '%Y-%m-%d').date())
    
        # if home team exists and in same season
        if game['HOME_TEAM'] in team_rats and int(team_rats[game['HOME_TEAM']][-1][0][0:4]) == int(date[0:4]):
            
            # get current home rating
            home_rat = team_rats[game['HOME_TEAM']][-1][1]
        
        # if home team exists and NOT in same season
        elif game['HOME_TEAM'] in team_rats:
            
            # get initial rating
            init_rat = get_init_rat(game['HOME_TEAM'], fbs_team_list)
            
            # reset home rating
            home_rat = retain_weight*(team_rats[game['HOME_TEAM']][-1][1]-init_rat) + init_rat
        
        # if NOT home team exists
        else:
            
            # get initial rating
            init_rat = get_init_rat(game['HOME_TEAM'], fbs_team_list)
            
            # append home team to dict
            team_rats[game['HOME_TEAM']] = [(date, init_rat, None, None)]
            home_rat = team_rats[game['HOME_TEAM']][-1][1]
        
        # if away team exists and in same season
        if game['AWAY_TEAM'] in team_rats and int(team_rats[game['AWAY_TEAM']][-1][0][0:4]) == int(date[0:4]):
            
            # get current home rating
            away_rat = team_rats[game['AWAY_TEAM']][-1][1]
        
        # if away team exists and NOT in same season
        elif game['AWAY_TEAM'] in team_rats:
            
            # get initial rating
            init_rat = get_init_rat(game['AWAY_TEAM'], fbs_team_list)
            
            # reset away rating
            away_rat = retain_weight*(team_rats[game['AWAY_TEAM']][-1][1]-init_rat) + init_rat
        
        # if NOT away team exists
        else:
            
            # get initial rating
            init_rat = get_init_rat(game['AWAY_TEAM'], fbs_team_list)
            
            # append away team to dict
            team_rats[game['AWAY_TEAM']] = [(date, init_rat, None, None)]
            away_rat = team_rats[game['AWAY_TEAM']][-1][1]
        
        # calc score margin from game
        margin = game['HOME_POINTS'] - game['AWAY_POINTS']
    
        # calc new ratings
        home_info, away_info = calc_new_rats(home_rat, away_rat, margin, K=perm[1], scaler=perm[2])
        home_rat_new, home_conf, home_act = home_info
        away_rat_new, away_conf, away_act = away_info
        
        # append new ratings to dict
        team_rats[game['HOME_TEAM']].append( (date, home_rat_new, home_conf, home_act) )
        team_rats[game['AWAY_TEAM']].append( (date, away_rat_new, away_conf, away_act) )
        
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
    
    # calc log loss & accuracy
    if True:
        
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
    
    if False:
        print("\n >>> perm: {}".format(perm))
        
        test_loss = log_loss(test_acts, test_preds)
        print("\n >>> Log Loss (Test Data): {}".format(test_loss))
        
        naive_test_loss = log_loss(naive_test_acts, naive_test_preds)
        print(" >>> Naive Log Loss Test Data): {}".format(naive_test_loss))
        
        test_acc = accuracy_score(test_acts, list(map(round,test_preds)))
        print("\n >>> Accuracy (Test Data): {}".format(test_acc))
        
        naive_test_acc = accuracy_score(naive_train_acts, list(map(round,naive_train_preds)))
        print(" >>> Naive Accuracy (Test Data): {}".format(naive_test_acc))
        
        print("\n")
