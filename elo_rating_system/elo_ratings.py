"""
College Football Data Analytics: Elo Ratings
Author: Trevor Cross
Last Updated: 06/21/22

Simuates NCAAF games using an Elo rating algorithm.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import support functions
import sys
from tqdm import tqdm
from datetime import datetime
from operator import itemgetter
from sklearn.metrics import log_loss, accuracy_score

# import toolbox functions
sys.path.insert(1, '/home/tjcross/CFD_demo/')
from toolbox import *

# ---------------------------
# ---Pull Game Information---
# ---------------------------

# connect to SF
conn = connect_to_SF()

# obtain game data
game_query = """
             select start_date, home_team, home_points, away_team, away_points from games
             where season >= 1920 and season <= 2019
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

# ------------------------
# ---Run Elo Simulation---
# ------------------------

# define reset weight
retain_weight = 1.00 ## retain_weight=1 => no reset, retain_weight=0 => reset to init_rat

# create dictionary to hold team Elo ratings
team_rats = dict()

# iterate through games
for game_num, game in tqdm(game_df.iterrows(), desc='Iterating games ', unit=' game', total=game_df.shape[0]):
    
    # parse current date
    date = str(datetime.strptime(game['START_DATE'][0:10], '%Y-%m-%d').date())

    # if home team exists and in same season
    if game['HOME_TEAM'] in team_rats and int(team_rats[game['HOME_TEAM']][-1][0][0:4]) == int(date[0:4]):
        
        # get current home rating
        home_rat = team_rats[game['HOME_TEAM']][-1][1]
    
    # if home team exists and NOT in same season
    elif game['HOME_TEAM'] in team_rats:
        
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
    home_info, away_info = calc_new_rats(home_rat, away_rat, margin, K=50)
    home_rat_new, home_conf, home_act = home_info
    away_rat_new, away_conf, away_act = away_info
    
    # append new ratings to dict
    team_rats[game['HOME_TEAM']].append( (date, home_rat_new, home_conf, home_act) )
    team_rats[game['AWAY_TEAM']].append( (date, away_rat_new, away_conf, away_act) )

# --------------------
# ---Evaluate Model---
# --------------------

# get predicted probabilities & actualized values
preds = []
acts = []
for key in team_rats:
    preds.extend(list(map(itemgetter(2), team_rats[key])))
    acts.extend(list(map(itemgetter(3), team_rats[key])))

# remove Nonetype objects
preds = [pred for pred in preds if pred != None]
acts = [act for act in acts if act != None]

# remove ties
while len(np.unique(acts)) > 2:
    # for some (dumb) reason, the below loop will not remove all 0.5's & related
    # indices the first go around. So it's ran until we get only the binary
    for index, value in enumerate(acts):
        if value == 0.5:
            del preds[index]
            del acts[index]

# split train and test data
train_preds, test_preds = preds[:int(0.7*len(preds))], preds[int(0.7*len(preds)):]
train_acts, test_acts = acts[:int(0.7*len(acts))], acts[int(0.7*len(acts)):]

# calc log loss & accuracy
if True:
    train_loss = log_loss(train_acts, train_preds)
    print("\n >>> Log Loss (Train Data): {}".format(train_loss))
    
    train_acc = accuracy_score(train_acts, list(map(round,train_preds)))
    print("\n >>> Accuracy (Train Data): {}".format(train_acc))

if False:
    test_loss = log_loss(test_acts, test_preds)
    print("\n >>> Log Loss (Test Data): {}".format(test_loss))
    
    test_acc = accuracy_score(test_acts, list(map(round,test_preds)))
    print("\n >>> Accuracy (Test Data): {}".format(test_acc))
