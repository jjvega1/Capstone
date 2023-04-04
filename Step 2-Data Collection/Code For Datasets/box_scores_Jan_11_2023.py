from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from googlesearch import search

# URL containing all the games on January 11, 2023
my_url = 'https://www.cbssports.com/college-basketball/scoreboard/FBS/20230111/'

uClient = uReq(my_url)  # open up connection
page_html = uClient.read()  # grab the page
uClient.close()  # We are done with this (for now)

# Parse html file
page_soup = soup(page_html, "html.parser")

# Extract all cells containing the names
cells = page_soup.findAll("td", {"class": "team team--collegebasketball"})

# Extract all cells containing

# Create a list of tuples to store these names
n = []

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


# Add teams and scores to list
for i in range(0, len(cells), 2):
    n.append((remove_it(cells[i].text), remove_it(cells[i+1].text)))

# Check to see if it worked
'''
for j in range(len(l)):
    print(l[j])
'''

# Now that we have our teams, we will now search for it on Google.
# We want to get the ESPN box score url using the teams as the query

# First here is an example of getting the box score
# url for the American vs Army Game
'''
query = F'{n[0][0]} vs {n[0][1]} Jan 11 2023 College Basketball'
for i in search(query, tld="co.in", num=5, stop=5, pause=2):
    if 'espn' in i:
        espn_url = i
        break

# Change tabs to get the desired stats
espn_url = espn_url.replace('game', 'matchup', 1)
print(espn_url)

# The usual
uClient_espn = uReq(espn_url)
espn_html = uClient_espn.read()
uClient_espn.close()
espn_soup = soup(espn_html, "html.parser")

# Get the points
p = []
espn_pts = espn_soup.findAll(
    "div", {"class": "Gamestrip__Score relative tc w-100 fw-heavy h2 clr-gray-01"})
p.append((int(espn_pts[0].text), int(espn_pts[1].text)))
print(p)

# Get all the stats
espn_stats = espn_soup.findAll("td", {"class": "Table__TD"})

# Then get the rebounds
r = []
for j in range(len(espn_stats)):
    if espn_stats[j].text == "Rebounds":
        r.append((int(espn_stats[j+1].text), int(espn_stats[j+2].text)))
        break

# Then get the assists
a = []
for k in range(len(espn_stats)):
    if espn_stats[k].text == "Assists":
        a.append((int(espn_stats[k+1].text), int(espn_stats[k+2].text)))
        break

# The way we will write this in the csv file is that
# our columns will be seperated by winner and loser
# we can check the winner by comparing who was more points
# and assign the appropiate stats to the columns
if p[0][0] > p[0][1]:  # First team won
    w = 0
    l = 1
else:  # Second team won
    w = 1
    l = 0

filename = "American_v_Army"
f = open(filename, "w")
headers = "Winner, Loser, w_pts, l_pts, w_reb, l_reb, w_ast, l_ast"
f.write(headers + "\n")
f.write(F'{n[0][w]}, {n[0][l]}, {p[0][w]}, {p[0][l]}, {r[0][w]}, {r[0][l]}, {a[0][w]}, {a[0][l]}')
f.close()
'''

# So now we do this for all teams on that day
filename = "Jan 11, 2023 Box Scores.csv"
f = open(filename, "w")
headers = "Winner, Loser, w_pts, l_pts, w_reb, l_reb, w_ast, l_ast"
f.write(headers + "\n")
p = []
r = []
a = []

for i in range(len(n)):
    query = F'{n[i][0]} vs {n[i][1]} Jan 11 2023 College Basketball'
    for j in search(query, tld="co.in", num=5, stop=5, pause=4):
        if 'espn' in j:
            espn_url = j
            break

    # change url to get right tab and load Beautiful Soup
    # sometimes, it will give me different tabs (gamecast, boxscore, play-by-play, or team stats)
    # what we really need is team stats tab
    if espn_url[45:49] == "game":
        espn_url = espn_url.replace('game', 'matchup', 1)
    elif espn_url[45:53] == "boxscore":
        espn_url = espn_url.replace('boxscore', 'matchup', 1)
    elif espn_url[45:55] == "playbyplay":
        espn_url = espn_url.replace('playbyplay', 'matchup', 1)

    uClient_espn = uReq(espn_url)
    espn_html = uClient_espn.read()
    uClient_espn.close()
    espn_soup = soup(espn_html, "html.parser")

    # points
    espn_pts = espn_soup.findAll(
        "div", {"class": "Gamestrip__Score relative tc w-100 fw-heavy h2 clr-gray-01"})
    p.append((int(espn_pts[0].text), int(espn_pts[1].text)))

    # get the rest of the stats
    espn_stats = espn_soup.findAll("td", {"class": "Table__TD"})

    # rebounds and assists
    for k in range(len(espn_stats)):
        if espn_stats[k].text == "Rebounds":
            r.append((int(espn_stats[k+1].text), int(espn_stats[k+2].text)))

        if espn_stats[k].text == "Assists":
            a.append((int(espn_stats[k+1].text), int(espn_stats[k+2].text)))

        if len(r) == len(p) and len(a) == len(p):
            break

    if p[i][0] > p[i][1]:  # First team won
        w = 0
        l = 1
    else:  # Second team won
        w = 1
        l = 0

    f.write(F'{n[i][w]},')
    f.write(F'{n[i][l]},')
    f.write(F'{p[i][w]},')
    f.write(F'{p[i][l]},')
    f.write(F'{r[i][w]},')
    f.write(F'{r[i][l]},')
    f.write(F'{a[i][w]},')
    f.write(F'{a[i][l]} \n')

f.close()
