"""
College Football Data Demo: Season Record Evaluation
Author: Trevor Cross
Last Updated: 06/29/22

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
from tqdm import tqdm
from datetime import datetime
from operator import itemgetter
from time import time

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

# define Elo rating algorithm parameters
K = 25
scaler = 350

# run season simulations
num_sims = 100
start_time = time()
with Pool(2) as p:
    team_rats_hot = list(p.starmap(run_season_sim, [(2000, game_df, fbs_team_list, team_rats, K, scaler)]*num_sims))
print("\n >>> Time Elapsed: {}".format(time()-start_time))

# # aggregate game statistics
# team_rats_mean = dict()
# for key_num, key in enumerate(team_rats_hot[0]):
#     mean_rat = np.mean([team_rats_hot[i][key][1] for i in range(len(team_rats_hot))])
#     mean_wins = np.mean([team_rats_hot[i][key][3] for i in range(len(team_rats_hot))])
#     team_rats_mean[key] = [(mean_rat, mean_wins)]