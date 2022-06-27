"""
College Football Data Demo: Generate Initial Elo Ratings
Author: Trevor Cross
Last Updated: 06/27/22

Produces a starting Elo rating for each team per season. Each season is saved
as a separate JSON file.
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
             where season >= 1969 and season < 2022
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