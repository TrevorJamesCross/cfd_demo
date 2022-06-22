"""
NCAAF Demo: Serverless Function
Author: Trevor Cross
Last Updated: 06/22/22

Define a serverless function to retrieve data from an API call. This script
is intended for use in Fivetran as a connector to Snowflake.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import requests
import requests

# ----------------------
# ---Define Functions---
# ----------------------

def make_request(url, api_key):
    
    # define headers
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer {}".format(api_key)}
    
    # return API call
    return requests.get(url, headers=headers)
    
def handler(request):
    
    # define request url
    year = 2021
    url = "https://api.collegefootballdata.com/recruiting/teams?year={}".format(year)
    
    # get credentials
    request_json = request.get_json()
    api_key = request_json['secrets']['api_key']
    
    # make API request
    response_json = make_request(url, api_key).json()
    
    # format json for fivetran
    final = {
        "insert": {
            "recruiting_rankings": response_json
            },
        "state": "state_0",
        "hasMore": False
        }
    
    # return final
    return final
