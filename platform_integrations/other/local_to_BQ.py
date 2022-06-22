"""
NCAAF Demo: Local Data to BigQuery
Author: Trevor Cross
Last Updated: 04/18/22

Create an algorithm to send local csv data to BigQuery. This script is NOT
assumed to run in its entirety, but iterively sub-section by sub-section.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import google libraries
from google.cloud import bigquery
from google.oauth2 import service_account

# import support libraries
import decimal
from os.path import join

# -----------------------
# ---Connect to Client---
# -----------------------

# define working directory path
working_dir_path = "/home/tjcross/python_dir/projects/ncaaf_demo"

# build credentials
key_file = #<key>
key_path = join(working_dir_path, key_file)

credentials = service_account.Credentials.from_service_account_file(key_path)

# define destination client
client = bigquery.Client(credentials=credentials)

# -----------------------
# ---Prepare Data File---
# -----------------------

# build table path
project_name = #<project_name>
dataset_name = #<dataset_name>
table_name = #<table_name>

table_id = project_name + "." + dataset_name + "." + table_name

# load local file
file_name = "game_boxscore_data.csv"
file_path = join(working_dir_path, file_name)

# -------------------------------
# ---Config & Send API Request---
# -------------------------------

# configure API request settings
job_config = bigquery.LoadJobConfig(
    schema = [bigquery.SchemaField('game_id', 'string', mode='Required'),
              bigquery.SchemaField('home_points','numeric'),
              bigquery.SchemaField('away_points','numeric')])

# upload data to BigQuery (API Request)
file_df = pd.read_csv(file_path)

file_df['game_id'] = np.arange(file_df.shape[0])
file_df['game_id'] = file_df['game_id'].astype(str)

file_df['home_points'] = file_df['home_points'].astype(str).map(decimal.Decimal) ## mapped to decimal type to patch error
file_df['away_points'] = file_df['away_points'].astype(str).map(decimal.Decimal) ## mapped to decimal type to patch error

job = client.load_table_from_dataframe(file_df, table_id, job_config=job_config)

# wait for job to complete
job.result()

# verify upload to BigQuery (API Request)
table = client.get_table(table_id)
print(">>> Loaded {} rows and {} columns to {}".format(
    table.num_rows, len(table.schema), table_id))
