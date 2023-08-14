import sys
import logging
import unittest
import pandas as pd
import numpy as np
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set up the logger
logging.basicConfig(filename='logs.log', filemode='w')
logger = logging.getLogger()

# Check for correct arguments
if len(sys.argv) == 1:
    logger.error("At least provide an argument that represents a year.")
elif len(sys.argv) == 3:
    logger.error("Please provide another teamID please.")
elif len(sys.argv) == 2:
    try:
        year_entered = int(sys.argv[1])
        if year_entered < 1985 or year_entered > 2022 or year_entered == 2020:
            logger.error(
                "Enter a valid year. Valid years are anything between 1985 and 2022 and NOT 2020.")
    except:
        logger.error("Please enter an integer for the year.")
elif len(sys.argv) == 4:
    try:
        year_entered = int(sys.argv[1])
        team_one = int(sys.argv[2])
        team_two = int(sys.argv[3])
        if team_one < 1101 or team_one > 1477 or team_two < 1101 or team_two > 1477:
            logger.error(
                "Please make sure you entered a valid TeamID. Valid TeamID's are any number from 1101 to 1477.")
        elif team_one == team_two:
            logger.warning("Comparing the same team?")
        elif year_entered < 1985 or year_entered > 2022 or year_entered == 2020:
            logger.error(
                "Enter a valid year. Valid years are anything between 1985 and 2022 and NOT 2020.")
    except:
        logger.error("Make Sure all your inputs are integers")

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

# Create the new row to cover the missing 2021's VCU forced forefit to Oregon St.
tourney_compact_pd.loc[len(tourney_compact_pd)] = [
    2021, 138, 1332, 64, 1433, 59, 'N', 0]

# Re-index it
tourney_compact_pd = tourney_compact_pd.sort_values(
    ['Season', 'DayNum', 'WTeamID']).reset_index()

# Delete that pesky index column
tourney_compact_pd = tourney_compact_pd.drop(columns=['index'])

c1 = tourney_compact_pd['Season'] == 2021
c2 = tourney_compact_pd['DayNum'] == 138
tourney_compact_pd[c1 & c2]

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

    return team_id in p6_list

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

# Get DataFrame of the Tourney Results for the given year


def getAppropiateTournamentDataFrame(year):
    tourney_year = tourney_compact_pd[tourney_compact_pd["Season"] == year]
    seeds_year = seeds_pd[seeds_pd["Season"] == year]

    return tourney_year, seeds_year

# PlayIn Winners


def getPlayInWinners(df, year):
    playin = []

    if year < 2001:
        return playin
    elif year >= 2011:
        for i in range(4):
            playin.append(df.iloc[i, 2])
    else:
        playin.append(df.iloc[0, 2])

    return playin

# Helper function to swap


def swap_it(m, x1, x2):
    m[x1], m[x2] = m[x2], m[x1]
    return m

# Get First Round Matchups for that given year for 2011 and later


def getMatchupsPost2011(df, year, playin):
    matchups = []
    k = 0

    # We will have to find the playin winner from each quadrant
    for i in range(8):
        if len(df.iloc[16 - i, 1]) == 4:
            if df.iloc[16 - i, 2] not in playin:
                k = 1
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[16 - i - k, 2]})

    k = 0

    for i in range(17, 25):
        if len(df.iloc[50 - i, 1]) == 4:
            if df.iloc[50 - i, 2] not in playin:
                k = 1
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[50 - i - k, 2]})

    k = 0

    for i in range(34, 42):
        if len(df.iloc[84 - i, 1]) == 4:
            if df.iloc[84 - i, 2] not in playin:
                k = 1
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[84 - i - k, 2]})

    k = 0

    for i in range(51, 59):
        if len(df.iloc[118 - i, 1]) == 4:
            if df.iloc[118 - i, 2] not in playin:
                k = 1
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[118 - i - k, 2]})

    for i in range(0, 4):
        matchups = swap_it(matchups, 8 * i + 1, 8 * i + 7)
        matchups = swap_it(matchups, 8 * i + 2, 8 * i + 4)

    return matchups

# Get First Round Matchups for that given year for before 2001


def getMatchupsPre2001(df, year):
    matchups = []

    # Since there is no playin, we can generate our matchups easily
    for i in range(8):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[15 - i, 2]})

    for i in range(16, 24):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[47 - i, 2]})

    for i in range(32, 40):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[79 - i, 2]})

    for i in range(48, 56):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 2], 'Team2ID': df.iloc[101 - i, 2]})

    for i in range(0, 4):
        matchups = swap_it(matchups, 8 * i + 1, 8 * i + 7)
        matchups = swap_it(matchups, 8 * i + 2, 8 * i + 4)

    return matchups

# Get First Round Matchups for that given year for 2001 - 2010


def getMatchups2001_2010(df, year, playin):
    matchups = []

    # For this one, we first find the PlayIn game
    for i in range(len(df)):
        if len(df.iloc[64 - i, 1]) == 4:
            s = 64 - i
            break

    df = df.reset_index()
    # We will delete the row of the loser
    if df.iloc[s, 1] not in playin:
        df1 = df.drop(s).reset_index()
    else:
        df1 = df.drop(s - 1).reset_index()

    # Then we proceed like the above function
    for i in range(8):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 3], 'Team2ID': df.iloc[15 - i, 3]})

    for i in range(16, 24):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 3], 'Team2ID': df.iloc[47 - i, 3]})

    for i in range(32, 40):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 3], 'Team2ID': df.iloc[79 - i, 3]})

    for i in range(48, 56):
        matchups.append(
            {'Round': 1, 'Team1ID': df.iloc[i, 3], 'Team2ID': df.iloc[101 - i, 3]})

    for i in range(0, 4):
        matchups = swap_it(matchups, 8 * i + 1, 8 * i + 7)
        matchups = swap_it(matchups, 8 * i + 2, 8 * i + 4)

    return matchups

# Create Logistic Regression Model and Train it. Return Model and Validation Accuracy Score


def getModel(xData, yData):
    X_train, X_val, y_train, y_val = train_test_split(
        xData, yData, test_size=.3, random_state=12)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_val = lr.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)

    return lr, val_acc

# Function to predict probability that Team 1 wins


def predictOutcome(team_data1, team_data2, model):
    diff = [a - b for a, b in zip(team_data1, team_data2)]
    diff.append(0)
    return model.predict([diff]), round(model.predict_proba([diff])[0][1], 4)

# Now we predict the tournament given the year


def predictTheTournament(year, matchups, model):
    # These variables will be used to increment the round appropiately
    r = 2
    threshold = 32

    # store the odd team
    odd_team = 0

    for i in range(63):
        # First get the IDs
        team1 = matchups[i]['Team1ID']
        team2 = matchups[i]['Team2ID']

        # Then get their data
        t1_data = getSeasonData(team1, 2022)
        t2_data = getSeasonData(team2, 2022)

        # Get 0 or 1 value by running the above function
        prob, p1 = predictOutcome(t1_data, t2_data, model)

        # Predict team
        if prob:
            matchups[i]['Predicted_Winner'] = team1
        else:
            matchups[i]['Predicted_Winner'] = team2

        matchups[i]['Probability'] = p1

        # Add a new row to the matchups once two games are complete until we have 63 games
        if len(matchups) < 64:
            if i % 2 == 1:
                matchups.append({'Round': r, 'Team1ID': odd_team,
                                'Team2ID': matchups[i]['Predicted_Winner']})
                odd_team = 0
            else:
                odd_team = team1

        if i == threshold:
            threshold += (32 / 2**(r - 1))
            r += 1

    return matchups

# Get the Actual Winners for each round


def getActualWinners(df, year):
    # First we need to know when to start our indexing
    if year < 2001:
        offset = 0
    elif year >= 2011:
        offset = 4
    else:
        offset = 1

    actual_winners = []
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    r6 = []

    # Make Round 1 list
    for i in range(offset, 32 + offset):
        r1.append(df.iloc[i, 2])

    # Round 2
    for i in range(32 + offset, 48 + offset):
        r2.append(df.iloc[i, 2])

    # Round 3
    for i in range(48 + offset, 56 + offset):
        r3.append(df.iloc[i, 2])

    # Round 4
    for i in range(56 + offset, 60 + offset):
        r4.append(df.iloc[i, 2])

    # Round 5
    for i in range(60 + offset, 62 + offset):
        r5.append(df.iloc[i, 2])

    # Round 6
    r6.append(df.iloc[62 + offset, 2])

    actual_winners.append(r1)
    actual_winners.append(r2)
    actual_winners.append(r3)
    actual_winners.append(r4)
    actual_winners.append(r5)
    actual_winners.append(r6)

    return actual_winners

# Are we correct?


def answers(true, pred):
    for i in range(len(pred)):
        predicted_winner = pred[i]['Predicted_Winner']
        r = pred[i]['Round']
        if predicted_winner in true[r - 1]:
            pred[i]['Correct'] = 1
        else:
            pred[i]['Correct'] = 0

    return pred

# Create DataFrame with the Predicted Winners and the Correct Column


def getResultsDataFrame(results, year):
    df = pd.DataFrame(results)
    df['Year'] = year
    return df

# How were the wins distributed by round?


def getRoundDist(df):
    correct_per_round = []
    for i in range(1, 7):
        c1 = df['Round'] == i
        cor = df['Correct'] == 1
        correct_per_round.append((i, len(df[c1 & cor])))

    return correct_per_round


def getDict(year, correct, round_dist, val_score):
    d = {}
    d['Year'] = year
    d['Correct'] = correct
    d['R1'] = round_dist[0][1]
    d['R2'] = round_dist[1][1]
    d['R3'] = round_dist[2][1]
    d['R4'] = round_dist[3][1]
    d['R5'] = round_dist[4][1]
    d['R6'] = round_dist[5][1]
    d['Correct %'] = round((correct / 63) * 100, 2)
    d['Val %'] = round(val_score * 100, 2)
    return d


def yearResult(year):
    # Create training/validation set
    xTrain, yTrain = createTrainingSet([year])

    # Get DataFrame of Tourney with the concerned year
    tourney_year, seeds_year = getAppropiateTournamentDataFrame(year)

    # Get PlayIn Winners
    playin = getPlayInWinners(tourney_year, year)

    # Generate matchups
    if year < 2001:
        matchups = getMatchupsPre2001(seeds_year, year)
    elif year >= 2011:
        matchups = getMatchupsPost2011(seeds_year, year, playin)
    else:
        matchups = getMatchups2001_2010(seeds_year, year, playin)

    # Get Trained model and Val Accuracy
    model, val_acc = getModel(xTrain, yTrain)

    # Get the Predicted Tournament Results
    results = predictTheTournament(year, matchups, model)

    # Get the actual results
    actual_results = getActualWinners(tourney_year, year)

    # Append our Predicted Tournament Results the Correct column
    results = answers(actual_results, results)

    # Get our DataFrame containing the Predicted Results and the Correct column
    results_pd = getResultsDataFrame(results, year)

    # Get Number of correct predictions
    num_correct = len(results_pd[results_pd['Correct'] == 1])

    # Finally Get Round Distribution
    round_dist = getRoundDist(results_pd)

    return getDict(year, num_correct, round_dist, val_acc), results, model

# Unit tests


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def testGetTeamID(self):
        self.assertEqual(1400, getTeamID('Texas'))
        self.assertFalse(1344 == getTeamID('Rice'))  # Rice's ID is 1349

    def testGetTeamName(self):
        self.assertEqual('Ball St', getTeamName(1123))
        self.assertFalse('Duke' == getTeamName(1194))  # 1194 is Fl Atlantic

    # For the Season Vector, go sports-reference for those stats
    def testGetRegSeasonWins(self):
        # Crazy that FDU had 4 wins before upsetting Purdue next year
        self.assertEqual(4, getRegSeasonWins(1192, 2022))
        # Baylor had 8 losses, but 27 wins
        self.assertFalse(8 == getRegSeasonWins(1124, 2017))

    def testGetPPG(self):
        self.assertTrue(abs(72.0 - getPPG(1181, 2023))
                        < 1)  # 1181 is Duke's ID
        # I thought Michigan St.'s defense was elite...
        self.assertFalse(abs(64.1 - getPPG(1292, 2016)) < 1)

    def testGetConferenceTourneyChampion(self):
        # UNC Ashville won the MVC tourney
        self.assertTrue(getTourneyConferenceChampion(1421, 2016))
        self.assertFalse(getTourneyConferenceChampion(
            1229, 2012))  # Illinois St lost to the winner


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# View total and individual results
if len(sys.argv) == 2:
    total_results, match_results, model = yearResult(year_entered)
    print("Total Evaluation:", total_results)
    for i in range(len(match_results)):
        it1 = getTeamName(match_results[i]['Team1ID'])
        it2 = getTeamName(match_results[i]['Team2ID'])
        it3 = match_results[i]['Probability']
        if match_results[i]['Correct']:
            corre = 'Yes'
        else:
            corre = 'No'
        if it1 == match_results[i]['Predicted_Winner']:
            print("Winner: %s Loser: %s Correct: %s Probability: %f" %
                  (it1, it2, corre, it3))
        else:
            print("Winner: %s Loser: %s Correct: %s Probability: %f" %
                  (it2, it1, corre, it3))

# Look at a game's probability
if len(sys.argv) == 4:
    total_results, match_results, model = yearResult(year_entered)
    t1_dat = getSeasonData(team_one, year_entered)
    t2_dat = getSeasonData(team_two, year_entered)
    it1 = getTeamName(team_one)
    it2 = getTeamName(team_two)
    prob, p1 = predictOutcome(t1_dat, t2_dat, model)
    p1 = round(p1 * 100, 2)
    print("Matchup: %s vs %s in %d" % (it1, it2, year_entered))
    print("%s has a %f percent chance to win." % (it1, p1))
