"""
College Football Data Demo: Season Record Evaluation
Author: Trevor Cross
Last Updated: 06/28/22

Simuates NCAAF games using an Elo rating algorithm to predict end season
records.
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

# import toolbox functions
repo_dir = join(expanduser('~'), 'CFD_demo')
sys.path.insert(1, repo_dir)
from toolbox import *

# ---------------
# ---Pull Data---
# ---------------

# define path to SF credentials
json_creds_path = join(expanduser('~'),'secrets/SF_creds.json')

# connect to SF
conn = connect_to_SF(json_creds_path)

# obtain game data
game_query = """
             select start_date, home_team, home_points, away_team, away_points from games
             where season >= 1970 and season < 2021
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

# define path to Elo rating history JSON file
file_path = join(expanduser('~'), 'CFD_demo/elo_rating_system/team_rating_hist.json')

# load JSON file as Python dictionary
team_rats = json_to_dict(file_path)

# ---------------------------
# ---Run Season Simulation---
# ---------------------------

# define seasons to simulate
seasons = [2000] #np.arange(2000, 2021, 1)

# create dictionary to record hot team ratings
team_rats_hot = dict()

# define retain weight
retain_weight = 0.85
K = 25
scaler = 350

# iterate seasons
for season in seasons:
    
    # filter game_df by season
    game_df_hot = game_df.loc[game_df['START_DATE'].str.startswith(str(season))]
    
    # filter game_df for previous seasons
    game_df_cold = game_df.loc[game_df['START_DATE'].str[:4].astype(int) < season]
    
    # iterate games
    for game_num, game in tqdm(game_df_hot.iterrows(), desc='Season {} '.format(season), unit=' game', total=game_df_hot.shape[0]):
        
        # parse current date
        date = str(datetime.strptime(game['START_DATE'][0:10], '%Y-%m-%d').date())
        
        # if home team exists
        if game['HOME_TEAM'] in team_rats_hot:
            
            # get current home rating
            home_rat = team_rats[game['HOME_TEAM']][-1][1]
        
        # if NOT home team exists
        else:
            
            # get initial rating
            init_rat = get_init_rat(game['HOME_TEAM'], fbs_team_list)
            
            # get starting rating for the season
            init_rat_hot = next(items[1] for items in team_rats[game['HOME_TEAM']] if items[0][:4]==str(season))
            
            # reset home rating
            home_rat = int(retain_weight*(init_rat_hot-init_rat) + init_rat)
            
            # append home team to dict
            team_rats_hot[game['HOME_TEAM']] = [(date, home_rat, None, None)]
            
        # if home team exists
        if game['AWAY_TEAM'] in team_rats_hot:
            
            # get current home rating
            home_rat = team_rats[game['AWAY_TEAM']][-1][1]
        
        # if NOT away team exists
        else:
            
            # get initial rating
            init_rat = get_init_rat(game['AWAY_TEAM'], fbs_team_list)
            
            # get starting rating for the season
            init_rat_hot = next(items[1] for items in team_rats[game['AWAY_TEAM']] if items[0][:4]==str(season))
            
            # reset away rating
            away_rat = int(retain_weight*(init_rat_hot-init_rat) + init_rat)
            
            # append away team to dict
            team_rats_hot[game['AWAY_TEAM']] = [(date, away_rat, None, None)]
        
        # sample game results
        home_info, away_info = sample_game_results(home_rat, away_rat, game_df_cold, K=K, scaler=scaler)
        home_rat_new, home_conf, home_act = home_info
        away_rat_new, away_conf, away_act = away_info
        
        # append new ratings to dict
        team_rats_hot[game['HOME_TEAM']].append( (date, home_rat_new, home_conf, home_act) )
        team_rats_hot[game['AWAY_TEAM']].append( (date, away_rat_new, away_conf, away_act) )
        
# ----------------------
# ---Evaluate Results---
# ----------------------