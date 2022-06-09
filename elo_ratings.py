"""
College Football Data Analytics: Elo Ratings
Author: Trevor Cross
Last Updated: 06/09/22

Simuates NCAAF games using an Elo rating algorithm.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import support functions
from tqdm import tqdm

# import toolbox functions
from toolbox import *

# remove warnings
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# ---Pull Game Information---
# ---------------------------

# connect to SF
conn = connect_to_SF()

# obtain game data
game_query = """select season, week, home_team, home_points, away_team, away_points from games"""
game_df = pd.read_sql(game_query, conn)

# ------------------------
# ---Run Elo Simulation---
# ------------------------

# define initial Elo rating
init_rat = 1500

# create dictionary to hold team Elo ratings
team_rats = dict()

# iterate through games
for game_num, game in tqdm(game_df.iterrows(), desc='Iterating games ', unit=' game', total=game_df.shape[0]):
    
    # get current rating for home team
    if game['HOME_TEAM'] in team_rats:
        home_rat = team_rats[game['HOME_TEAM']][-1]

    else:
        team_rats[game['HOME_TEAM']] = [init_rat]
        home_rat = team_rats[game['HOME_TEAM']][-1]
    
    # get current rating for away team
    if game['AWAY_TEAM'] in team_rats:
        away_rat = team_rats[game['AWAY_TEAM']][-1]

    else:
        team_rats[game['AWAY_TEAM']] = [init_rat]
        away_rat = team_rats[game['AWAY_TEAM']][-1]
        
    # calculate score margin from game
    margin = game['HOME_POINTS'] - game['AWAY_POINTS']

    # get new ratings
    home_rat_new, away_rat_new = calc_new_rats(home_rat, away_rat, margin)

    # append new ratings to dict
    team_rats[game['HOME_TEAM']].append(home_rat_new)
    team_rats[game['AWAY_TEAM']].append(away_rat_new)
    
# ------------------
# ---Plot Results---
# ------------------

# define teams to plot
team_keys = ['Wisconsin', 'Minnesota', 'Rutgers', 'Penn State']

# plot ratings against games played
plot_rats(team_rats, team_keys)
