from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from googlesearch import search
import random
import time
import sys
import os

# First enter season
season = "2011-12"

path = F"/Users/janjacob/Step 5: Data Wrangling and Exploration/{season}_bs_by_date"

# If directory for that season does not exist
if not os.path.exists(F"{path}/{season}_tracker.txt"):
    sys.exit(F"Please run file_creator.py on the {season} season first.")

# Find the next empty file and get that date
empty = True
for file in os.listdir(path):
    if os.stat(F"{path}/{file}").st_size == 0:
        curr_date = str(file)[0:8]
        year = curr_date[0:4]
        month = curr_date[4:6]
        day = curr_date[6:]
        empty = False
        break

if empty:
    sys.exit("All files have been filled.")

with open(F"{path}/{curr_date}_bs.csv", "a") as f:
    # Get all games on that day using CBS
    url_cbs = F'https://www.cbssports.com/college-basketball/scoreboard/FBS/{curr_date}/'
    uClient_cbs = uReq(url_cbs)
    page_html_cbs = uClient_cbs.read()
    uClient_cbs.close()
    page_soup_cbs = soup(page_html_cbs, "html.parser")
    cells = page_soup_cbs.findAll(
        "td", {"class": "team team--collegebasketball"})
    if len(cells) == 0:  # No games happened
        f.write("No games were played on this date.")
        sys.exit("No games were played on this date.")
    matchups = []

    # Add the points while we are at it
    pts_page = page_soup_cbs.findAll("td", {"class": ""})

    # A function to remove all numbers and dashes
    def remove_it(x):
        x = x.replace("1", "")
        x = x.replace("2", "")
        x = x.replace("3", "")
        x = x.replace("4", "")
        x = x.replace("5", "")
        x = x.replace("6", "")
        x = x.replace("7", "")
        x = x.replace("8", "")
        x = x.replace("9", "")
        x = x.replace("0", "")
        x = x.replace("-", "")
        x = x.strip()
        return x

    for j in range(0, len(cells), 2):
        matchups.append((remove_it(cells[j].text), remove_it(
            cells[j+1].text), int(pts_page[j].text), int(pts_page[j+1].text)))

    # Get all game data for each matchup
    print(F"Running this date: {curr_date}")
    print(F"We have {len(matchups)} matchups this day")

    for i in range(len(matchups)):
        print(F"Matchup {i}: {matchups[i]}")
        # Use ESPN to get our data
        query = F"{matchups[i][0]} vs {matchups[i][1]} college basketball espn {month}/{day}/{year}"
        for website in search(query, tld="co.in", num=5, stop=5, pause=4):
            r = random.randint(60, 120)
            time.sleep(r)
            print(F"Waited {r} seconds...")
            if 'espn' in website:
                espn_url = website
                break
        # If we did not find the url we wanted
        if espn_url == "":
            continue

        if espn_url.count("game") == 2:
            espn_url = espn_url.replace('game', 'matchup', 1)
        elif espn_url.count("boxscore") == 1:
            espn_url = espn_url.replace('boxscore', 'matchup', 1)
        elif espn_url.count("playbyplay") == 1:
            espn_url = espn_url.replace('playbyplay', 'matchup', 1)
        elif espn_url.count("recap") == 1:
            espn_url = espn_url.replace('recap', 'matchup', 1)

        uClient_espn = uReq(espn_url)
        html_espn = uClient_espn.read()
        uClient_espn.close()
        espn_soup = soup(html_espn, "html.parser")
        espn_stats = espn_soup.findAll("td", {"class": "Table__TD"})

        # Create tuples for each stat
        fg = "n1"
        fgp = -1
        fg3 = "n1"
        fgp3 = -1
        ft = "n1"
        ftp = -1
        reb = -1
        oreb = -1
        dreb = -1
        ast = -1
        stl = -1
        blk = -1
        to = -1
        pf = -1
        tf = -1
        ff = -1
        ll = -1

        # Get the points
        espn_pts = espn_soup.findAll(
            "div", {"class": "Gamestrip__Score relative tc w-100 fw-heavy h2 clr-gray-01"})

        # Find out which team won
        if matchups[i][2] > matchups[i][3]:
            win = 1
            lose = 2
            f.write(
                F"{matchups[i][0]},{matchups[i][1]},{matchups[i][2]},{matchups[i][3]},")
        else:  # If the home team won
            win = 2
            lose = 1
            f.write(
                F"{matchups[i][1]},{matchups[i][0]},{matchups[i][3]},{matchups[i][2]},")

        found = [False for i in range(17)]
        for j in range(len(espn_stats)):
            if espn_stats[j].text == "FG":
                fg = (espn_stats[j+win].text, espn_stats[j+lose].text)
                found[0] = True
            elif espn_stats[j].text == "Field Goal %":
                fgp = (float(espn_stats[j+win].text),
                       float(espn_stats[j+lose].text))
                found[1] = True
            elif espn_stats[j].text == "3PT":
                fg3 = (espn_stats[j+win].text, espn_stats[j+lose].text)
                found[2] = True
            elif espn_stats[j].text == "Three Point %":
                fgp3 = (float(espn_stats[j+win].text),
                        float(espn_stats[j+lose].text))
                found[3] = True
            elif espn_stats[j].text == "FT":
                ft = (espn_stats[j+win].text, espn_stats[j+lose].text)
                found[4] = True
            elif espn_stats[j].text == "Free Throw %":
                ftp = (float(espn_stats[j+win].text),
                       float(espn_stats[j+lose].text))
                found[5] = True
            elif espn_stats[j].text == "Rebounds":
                reb = (int(espn_stats[j+win].text),
                       int(espn_stats[j+lose].text))
                found[6] = True
            elif espn_stats[j].text == "Offensive Rebounds":
                oreb = (int(espn_stats[j+win].text),
                        int(espn_stats[j+lose].text))
                found[7] = True
            elif espn_stats[j].text == "Defensive Rebounds":
                dreb = (int(espn_stats[j+win].text),
                        int(espn_stats[j+lose].text))
                found[8] = True
            elif espn_stats[j].text == "Assists":
                ast = (int(espn_stats[j+win].text),
                       int(espn_stats[j+lose].text))
                found[9] = True
            elif espn_stats[j].text == "Steals":
                stl = (int(espn_stats[j+win].text),
                       int(espn_stats[j+lose].text))
                found[10] = True
            elif espn_stats[j].text == "Blocks":
                blk = (int(espn_stats[j+win].text),
                       int(espn_stats[j+lose].text))
                found[11] = True
            elif espn_stats[j].text == "Total Turnovers":
                to = (int(espn_stats[j+win].text),
                      int(espn_stats[j+lose].text))
                found[12] = True
            elif espn_stats[j].text == "Fouls":
                pf = (int(espn_stats[j+win].text),
                      int(espn_stats[j+lose].text))
                found[13] = True
            elif espn_stats[j].text == "Technical Fouls":
                tf = (int(espn_stats[j+win].text),
                      int(espn_stats[j+lose].text))
                found[14] = True
            elif espn_stats[j].text == "Flagrant Fouls":
                ff = (int(espn_stats[j+win].text),
                      int(espn_stats[j+lose].text))
                found[15] = True
            elif espn_stats[j].text == "Largest Lead":
                ll = (int(espn_stats[j+win].text),
                      int(espn_stats[j+lose].text))
                found[16] = True

            if all(found):
                break

        # Write it in the file
        if not fg == "n1":
            f.write(F"{fg[0]},{fg[1]},")
        else:
            f.write(",,")
        if not fgp == -1:
            f.write(F"{fgp[0]},{fgp[1]},")
        else:
            f.write(",,")
        if not fg3 == "n1":
            f.write(F"{fg3[0]},{fg3[1]},")
        else:
            f.write(",,")
        if not fgp3 == -1:
            f.write(F"{fgp3[0]},{fgp3[1]},")
        else:
            f.write(",,")
        if not ft == "n1":
            f.write(F"{ft[0]},{ft[1]},")
        else:
            f.write(",,")
        if not ftp == -1:
            f.write(F"{ftp[0]},{ftp[1]},")
        else:
            f.write(",,")
        if not reb == -1:
            f.write(F"{reb[0]},{reb[1]},")
        else:
            f.write(",,")
        if not oreb == -1:
            f.write(F"{oreb[0]},{oreb[1]},")
        else:
            f.write(",,")
        if not dreb == -1:
            f.write(F"{dreb[0]},{dreb[1]},")
        else:
            f.write(",,")
        if not ast == -1:
            f.write(F"{ast[0]},{ast[1]},")
        else:
            f.write(",,")
        if not stl == -1:
            f.write(F"{stl[0]},{stl[1]},")
        else:
            f.write(",,")
        if not blk == -1:
            f.write(F"{blk[0]},{blk[1]},")
        else:
            f.write(",,")
        if not to == -1:
            f.write(F"{to[0]},{to[1]},")
        else:
            f.write(",,")
        if not pf == -1:
            f.write(F"{pf[0]},{pf[1]},")
        else:
            f.write(",,")
        if not tf == -1:
            f.write(F"{tf[0]},{tf[1]},")
        else:
            f.write(",,")
        if not ff == -1:
            f.write(F"{ff[0]},{ff[1]},")
        else:
            f.write(",,")
        if not ll == -1:
            f.write(F"{ll[0]},{ll[1]} \n")
        else:
            f.write(", \n")
