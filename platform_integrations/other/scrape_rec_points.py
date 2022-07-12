"""
College Football Data Demo: Scrape Recruitment Scores
Author: Trevor Cross
Last Updated: 07/11/22

Scrapes latest recruitment scores from 247 sports.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import scraping libraries
import requests as req
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# import support libraries
import time
from os.path import expanduser, join

# --------------------
# ---Access Webpage---
# --------------------

# define webdriver options
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')

# define driver
chromedriver_path = join(expanduser('~'), 'drivers/chromedriver')
driver = webdriver.Chrome(chromedriver_path, options=options)
driver.maximize_window()

# access webpage
url = "https://247sports.com/Season/2022-Football/CompositeTeamRankings/"
driver.get(url)

# define base xpath to "Load More" button (note curly brackets)
base_xpath = '/html/body/section[1]/section/div/section/section/div/ul/li[{}]/a'

# click "Load More" button while it exists
teams_per_it = 50
max_it = 5
it = 1
while it <= max_it:
    try:        
        new_xpath = base_xpath.format(str( teams_per_it*it+2 ))
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, new_xpath))).click()
        print("\n >>> Clicked 'Load More'")
        
        it += 1
        
    except TimeoutException:
        print("\n >>> Timed Out")
        break

# get page source
page_source = driver.page_source

# quit driver
driver.quit()

# -------------------------------
# ---Scrape Recruitment Scores---
# -------------------------------

# obtain soup from page_source
soup = bs(page_source, 'html.parser')

# obtain team list from soup
team_list = []
for team_soup in soup.find_all('a', class_='rankings-page__name-link'):
    team_list.append(team_soup.get_text().strip())
    
# obtain points list from soup
points_list = []
for points_soup in soup.find_all('a', class_='number'):
    points_list.append(points_soup.get_text().strip())
points_list = list(map(float, points_list))

# create pandas DataFrame
rec_dict = {"YEAR":[2022]*(len(team_list)),
            "RANK":np.arange(1,len(team_list)+1),
            "TEAM":team_list,
            "POINTS":points_list}

rec_df = pd.DataFrame(data=rec_dict)