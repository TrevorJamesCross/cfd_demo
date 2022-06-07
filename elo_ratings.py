"""
College Football Data Analytics: Elo Ratings
Author: Trevor Cross
Last Updated: 05/25/22

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
# ---Pull Team Information---
# ---------------------------

# connect to SF
conn = connect_to_SF()

# obtain team names
team_query = "select school from teams"
team_df = pd.read_sql(team_query,conn)

# add column with base Elo ratings
init_rat = 1500
team_df['ELO'] = [ init_rat for _ in range(len(team_df)) ]


# ---------------------------
# ---Pull Game Information---
# ---------------------------

# obtain game data
game_query = """select home_team, home_points, away_team, away_points from games
            where season>=2015"""
game_df = pd.read_sql(game_query,conn)

# ------------------------
# ---Run Elo Simulation---
# ------------------------

# iterate through games
for game_num, game in tqdm(game_df.iterrows(), desc='Running Elo Sim', unit=' games', total=game_df.shape[0]):
    
    # get home & away initial elo ratings
    home_rat = int(team_df.loc[team_df['SCHOOL']==game['HOME_TEAM'], 'ELO'])
    away_rat = int(team_df.loc[team_df['SCHOOL']==game['AWAY_TEAM'], 'ELO'])
    
    # get margin of victory
    margin = game['HOME_POINTS'] - game['AWAY_POINTS']
    
    # calc new ratings
    home_rat_new, away_rat_new = calc_new_elo(home_rat, away_rat, margin)
    
    # replace ratings
    team_df.loc[team_df['SCHOOL']==game['HOME_TEAM'], 'ELO'] = home_rat_new
    team_df.loc[team_df['SCHOOL']==game['AWAY_TEAM'], 'ELO'] = away_rat_new

# close SF connection
conn.close()