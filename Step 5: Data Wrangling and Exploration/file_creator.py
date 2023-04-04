import os
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from dateutil import parser
import pandas as pd

# Specify the season first
# ex: If we wanted the 2011-12 season, season = "2011-12"
season = "2011-12"

# Create directory to store all these files (if it does not exist yet)
path = F"/Users/janjacob/Step 5: Data Wrangling and Exploration/{season}_bs_by_date"
if not os.path.exists(path):
    os.mkdir(F"{season}_bs_by_date")

# Get the start and end dates of the season
url_wiki = F'https://en.wikipedia.org/wiki/{season}_NCAA_Division_I_men%27s_basketball_season'
uClient = uReq(url_wiki)
page_html = uClient.read()
uClient.close()
page_soup = soup(page_html, "html.parser")
wiki_box = page_soup.findAll("td", {"class": "infobox-data"})
dates = wiki_box[1].text.split(" – ")

# Create file to keep track of finished dates (if it does not exist yet)
new_path = F"{path}/{season}_tracker"
if not os.path.exists(new_path):
    tracker = F"{season}_tracker"
    with open(os.path.join(path, F"{tracker}.txt"), "w") as file:
        file.write(F"Season: {season} \n")
        file.write(F"Start: {dates[0]} \n")
        file.write(F"End: {dates[1]} \n")

# Finally create separate date files
dr = pd.date_range(dates[0], dates[1])
dr = dr.strftime("%Y%m%d")
for i in range(len(dr)):
    if not os.path.exists(F"{path}/{dr[i]}_bs.csv"):
        f = open(F"{path}/{dr[i]}_bs.csv", "w")
        f.close()
