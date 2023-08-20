import pandas as pd
import numpy as np
import collections

# Regular Season Results since 1985 (Only includes who won and the points)
reg_season_compact_pd = pd.read_csv('Data/MRegularSeasonCompactResults.csv')

# Regular S eason Results since 2003 but includes useful stats like Rebounds, Assists, etc.
reg_season_detailed_pd = pd.read_csv('Data/MRegularSeasonDetailedResults.csv')

# List of teams who are/were in Division I along with their ID
teams_pd = pd.read_csv('Data/MTeams.csv')

# Like the first DataFrame but for the tournament only
tourney_compact_pd = pd.read_csv('Data/MNCAATourneyCompactResults.csv')

# The Conference Tourney Detailed Results since 2003
conference_tourney_results_pd = pd.read_csv('Data/MConferenceTourneyGames.csv')

# List of Teams along with their Conferences and ID per Year
conferences_pd = pd.read_csv('Data/MTeamConferences.csv')

# Seeds
seeds_pd = pd.read_csv('Data/MNCAATourneySeeds.csv')

# List of Conference Tourney Winners since 2001
l = []
for i in range(len(conference_tourney_results_pd) - 1):
    if conference_tourney_results_pd.iloc[i, 1] != conference_tourney_results_pd.iloc[i + 1, 1]:
        season = conference_tourney_results_pd.iloc[i, 0]
        conference = conference_tourney_results_pd.iloc[i, 1]
        winner = conference_tourney_results_pd.iloc[i, 3]
        l.append({'Season': season, 'Conference': conference, 'Winner': winner})

conference_tourney_winners_pd = pd.DataFrame(l)

# Get ID given Name


def getTeamID(name):
    return teams_pd[teams_pd['TeamName'] == name].values[0][0]

# Get Name given ID


def getTeamName(team_id):
    return teams_pd[teams_pd['TeamID'] == team_id].values[0][1]

# How many wins did a Team win in a given Season


def getRegSeasonWins(team_id, year):
    c1 = reg_season_compact_pd['WTeamID'] == team_id
    c2 = reg_season_compact_pd['Season'] == year
    return len(reg_season_compact_pd[c1 & c2])

# What was a team's Points per Game in a given Season


def getPPG(team_id, year):
    ppg = 0
    c1 = reg_season_compact_pd['WTeamID'] == team_id
    c2 = reg_season_compact_pd['Season'] == year
    c3 = reg_season_compact_pd['LTeamID'] == team_id
    gamesWon = reg_season_compact_pd[c1 & c2]
    ppg = gamesWon['WScore'].sum()
    gamesLost = reg_season_compact_pd[c2 & c3]
    ppg += gamesLost['LScore'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    ppg /= total_games
    return round(ppg, 2)

# In a given season, how many points did a given team give up per game


def getOPPG(team_id, year):
    oppg = 0
    c1 = reg_season_compact_pd['WTeamID'] == team_id
    c2 = reg_season_compact_pd['Season'] == year
    c3 = reg_season_compact_pd['LTeamID'] == team_id
    gamesWon = reg_season_compact_pd[c1 & c2]
    oppg = gamesWon['LScore'].sum()
    gamesLost = reg_season_compact_pd[c2 & c3]
    oppg += gamesLost['WScore'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    oppg /= total_games
    return round(oppg, 2)


# Set conditions for Power 6 Conference
acc = conferences_pd['ConfAbbrev'] == 'acc'
big12 = conferences_pd['ConfAbbrev'] == 'big_twelve'
bigeast = conferences_pd['ConfAbbrev'] == 'big_east'
big10 = conferences_pd['ConfAbbrev'] == 'big_ten'
pac12 = conferences_pd['ConfAbbrev'] == 'pac_twelve'
sec = conferences_pd['ConfAbbrev'] == 'sec'

# Make Dataframe where it only contains Power 6 teams
p6 = conferences_pd[acc | big12 | bigeast | big10 | pac12 | sec]


def getPower6(team_id, year):
    # Filter out the Dataframe for the appropiate year
    c1 = conferences_pd['Season'] == year

    # Get the list of TeamID's that sufficies all the conditions
    p6_list = list(p6[c1]['TeamID'])

    if team_id in p6_list:
        return 1
    else:
        return 0

# How many three's did a team make per game in a given season


def get3PT(team_id, year):
    if year < 2003:
        return 0
    threes = 0
    c1 = reg_season_detailed_pd['WTeamID'] == team_id
    c2 = reg_season_detailed_pd['Season'] == year
    c3 = reg_season_detailed_pd['LTeamID'] == team_id
    gamesWon = reg_season_detailed_pd[c1 & c2]
    threes = gamesWon['WFGM3'].sum()
    gamesLost = reg_season_detailed_pd[c2 & c3]
    threes += gamesLost['LFGM3'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    threes /= total_games
    return round(threes, 2)

# How many turnovers did a team make per game in a given season


def getTO(team_id, year):
    if year < 2003:
        return 0
    to = 0
    c1 = reg_season_detailed_pd['WTeamID'] == team_id
    c2 = reg_season_detailed_pd['Season'] == year
    c3 = reg_season_detailed_pd['LTeamID'] == team_id
    gamesWon = reg_season_detailed_pd[c1 & c2]
    to = gamesWon['WTO'].sum()
    gamesLost = reg_season_detailed_pd[c2 & c3]
    to += gamesLost['LTO'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    to /= total_games
    return round(to, 2)

# How many Assists did a team make per game


def getAST(team_id, year):
    if year < 2003:
        return 0
    ast = 0
    c1 = reg_season_detailed_pd['WTeamID'] == team_id
    c2 = reg_season_detailed_pd['Season'] == year
    c3 = reg_season_detailed_pd['LTeamID'] == team_id
    gamesWon = reg_season_detailed_pd[c1 & c2]
    ast = gamesWon['WAst'].sum()
    gamesLost = reg_season_detailed_pd[c2 & c3]
    ast += gamesLost['LAst'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    ast /= total_games
    return round(ast, 2)

# Easy way to get a team's conference in a given year


def getConference(team_id, year):
    c1 = conferences_pd['TeamID'] == team_id
    c2 = conferences_pd['Season'] == year
    c3 = conferences_pd[c1 & c2]
    if len(c3) == 0:
        return conferences_pd[c1].values[0][2]
    return c3['ConfAbbrev'].values[0]

# Determine if a team was the conference champion in their division in a given year


def getTourneyConferenceChampion(team_id, year):
    if year < 2001:
        return 0
    conf = getConference(team_id, year)
    c1 = conference_tourney_winners_pd['Season'] == year
    c2 = conference_tourney_winners_pd['Conference'] == conf
    if len(conference_tourney_winners_pd[c1 & c2]) == 0:
        return 0
    if team_id == conference_tourney_winners_pd[c1 & c2]['Winner'].values[0]:
        return 1
    else:
        return 0

# Get the seed of the team in a given year


def getSeed(team_id, year):
    c1 = seeds_pd['TeamID'] == team_id
    c2 = seeds_pd['Season'] == year
    if len(seeds_pd[c1 & c2]) == 0:
        return 0
    return int(seeds_pd[c1 & c2]['Seed'].values[0][1:3])

# Get rebounds per game in a given year


def getRPG(team_id, year):
    if year < 2003:
        return 0
    reb = 0
    c1 = reg_season_detailed_pd['WTeamID'] == team_id
    c2 = reg_season_detailed_pd['Season'] == year
    c3 = reg_season_detailed_pd['LTeamID'] == team_id
    gamesWon = reg_season_detailed_pd[c1 & c2]
    reb = gamesWon['WOR'].sum()
    reb += gamesWon['WDR'].sum()
    gamesLost = reg_season_detailed_pd[c2 & c3]
    reb += gamesLost['LOR'].sum()
    reb += gamesLost['LDR'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    reb /= total_games
    return round(reb, 2)

# Steals per game


def getSTL(team_id, year):
    if year < 2003:
        return 0
    stl = 0
    c1 = reg_season_detailed_pd['WTeamID'] == team_id
    c2 = reg_season_detailed_pd['Season'] == year
    c3 = reg_season_detailed_pd['LTeamID'] == team_id
    gamesWon = reg_season_detailed_pd[c1 & c2]
    stl = gamesWon['WStl'].sum()
    gamesLost = reg_season_detailed_pd[c2 & c3]
    stl += gamesLost['LStl'].sum()
    total_games = len(gamesWon) + len(gamesLost)
    stl /= total_games
    return round(stl, 2)

# How many times did a team appear in the tournament as of 2022


def getNumOfAppearances(team_id):
    return len(seeds_pd[seeds_pd['TeamID'] == team_id])

# Helper function for below


def getHomeStat(row):
    if (row == 'H'):
        home = 1
    if (row == 'A'):
        home = -1
    if (row == 'N'):
        home = 0
    return home

# Build the vector


def getSeasonData(team_id, year):
    # Check first if the team was Division 1 at the time
    c1 = teams_pd[teams_pd['TeamID'] ==
                  team_id]['FirstD1Season'].values[0] <= year
    c2 = teams_pd[teams_pd['TeamID'] ==
                  team_id]['LastD1Season'].values[0] >= year
    if ~c1 or ~c2:
        return []
    return [getRegSeasonWins(team_id, year),
            getPPG(team_id, year),
            getOPPG(team_id, year),
            getPower6(team_id, year),
            get3PT(team_id, year),
            getTO(team_id, year),
            getAST(team_id, year),
            getTourneyConferenceChampion(team_id, year),
            getSeed(team_id, year),
            getRPG(team_id, year),
            getSTL(team_id, year),
            getNumOfAppearances(team_id)]

# Build vectors for every team in a given season


def createSeasonDict(year):
    seasonDictionary = collections.defaultdict(list)
    for team in teams_pd['TeamName'].tolist():
        team_id = teams_pd[teams_pd['TeamName'] == team].values[0][0]
        team_vector = getSeasonData(team_id, year)
        seasonDictionary[team_id] = team_vector
    return seasonDictionary

# Basically run the function directly above this one on a set of years


def createTrainingSet(years):
    totalNumGames = 0
    for year in years:
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        totalNumGames += len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        totalNumGames += len(tourney.index)
    # Just choosing a random team and seeing the dimensionality of the vector
    numFeatures = len(getSeasonData(1181, 2012))
    xTrain = np.zeros((totalNumGames, numFeatures + 1))
    yTrain = np.zeros((totalNumGames))
    indexCounter = 0
    for year in years:
        team_vectors = createSeasonDict(year)
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        numGamesInSeason = len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        numGamesInSeason += len(tourney.index)
        xTrainSeason = np.zeros((numGamesInSeason, numFeatures + 1))
        yTrainSeason = np.zeros((numGamesInSeason))
        counter = 0
        for index, row in season.iterrows():
            w_team = row['WTeamID']
            w_vector = team_vectors[w_team]
            l_team = row['LTeamID']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = getHomeStat(row['WLoc'])
            if (counter % 2 == 0):
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [-p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        for index, row in tourney.iterrows():
            w_team = row['WTeamID']
            w_vector = team_vectors[w_team]
            l_team = row['LTeamID']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = 0  # All tournament games are neutral
            if (counter % 2 == 0):
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [-p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        xTrain[indexCounter:numGamesInSeason+indexCounter] = xTrainSeason
        yTrain[indexCounter:numGamesInSeason+indexCounter] = yTrainSeason
        indexCounter += numGamesInSeason
    return xTrain, yTrain


x_data, y_data = createTrainingSet([2023])
x_df = pd.DataFrame(x_data, columns=['Wins', 'PPG', 'OPPG', 'P6',
                    '3PT', 'TO', 'AST', 'ConfChamp', 'Seed', 'RPG', 'STL', 'NumApp', 'Home'])

y_df = pd.DataFrame(y_data, columns=['Team 1 Win?'])
train_df = pd.concat([x_df, y_df], axis=1)

train_df.to_csv('train_2023.csv', header=True, index=False)
