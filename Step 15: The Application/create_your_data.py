import sys
import logging
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set up the logger
logging.basicConfig(filename='logs.log', filemode='w')
logger = logging.getLogger()

# If no arguments were passed, we will run your file with your pairings
if len(sys.argv) == 1:
    your_matchups = pd.read_csv('list_of_matchups.csv')
    matchup_list = []
    for i in range(len(your_matchups)):
        matchup_list.append(
            [your_matchups.iloc[i, 0], your_matchups.iloc[i, 1]])
# If the two team names are passed
elif len(sys.argv) == 3:
    team1 = sys.argv[1]
    team2 = sys.argv[2]
# You ran it wrong
else:
    logger.error("Please refer to the README on how to run this file.")
    sys.exit()

# Regular Season Results (Only includes who won and the points)
reg_season_compact_pd = pd.read_csv('Data/MRegularSeasonCompactResults.csv')

# Regular Season Results but includes useful stats like Rebounds, Assists, etc.
reg_season_detailed_pd = pd.read_csv('Data/MRegularSeasonDetailedResults.csv')

# List of teams who are/were in Division I along with their ID
teams_pd = pd.read_csv('Data/MTeams.csv')

# Like the first DataFrame but for the tournament only
tourney_compact_pd = pd.read_csv('Data/MNCAATourneyCompactResults.csv')

# The Conference Tourney Detailed Results
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
            getNumOfAppearances(team_id),
            0]

# Function to predict probability that Team 1 wins


def getOurNums(team_data1, team_data2):
    return [a - b for a, b in zip(team_data1, team_data2)]


# If we passed in 2 teams as our arguments
if len(sys.argv) == 3:
    # Will be True if we spot a mistake in the First Team's Name
    b = False
    try:
        teamID1 = getTeamID(sys.argv[1])
    except:
        b = True
        logger.error(
            F'{sys.argv[1]} is spelled incorrectly. Please go to the MTeams.csv file to find the correct spelling.')

    try:
        teamID2 = getTeamID(sys.argv[2])
    except:
        logger.error(
            F'{sys.argv[2]} is spelled incorrectly. Please go to the MTeams.csv file to find the correct spelling.')
        sys.exit()

    if b:
        sys.exit()

    t1_data = getSeasonData(teamID1, 2023)
    t2_data = getSeasonData(teamID2, 2023)
    the_nums = getOurNums(t1_data, t2_data)

    print("\n Team1:", sys.argv[1])
    print("Team2:", sys.argv[2], '\n')

    print('Wins:', the_nums[0])
    print('PPG:', the_nums[1])
    print('OPPG:', the_nums[2])
    print('AST:', the_nums[6])
    print('Seed:', the_nums[8])
    print('RPG:', the_nums[9])
    print('TO:', the_nums[5])
    print('ConfChamp:', the_nums[7])
    print('NumApp:', the_nums[11])
    print('P6:', the_nums[3])
    print('3PT:', the_nums[4])
    print('STL:', the_nums[10])
    print('Home:', the_nums[12], '\n')

    print(
        F"If the result is over 50%, then {sys.argv[1]} is the predicted winner.")
    print(
        F"If the result is under 50%, then {sys.argv[2]} is the predicted winner. \n")
# We filled in our matchups in the csv file, so we passed in no arguments
else:
    matchup_data = pd.read_csv(
        'list_of_matchups.csv', names=['Team1', 'Team2'])
    if len(matchup_data) == 0:
        logger.error("You did not even put anything in there...")
        sys.exit()
    if matchup_data.isnull().values.any():
        logger.error(
            "Please make sure there are two teams separated by 1 comma per row.")
        sys.exit()

    nums = []

    # Again, to check for wrong spellings
    b = False
    for i in range(len(matchup_data)):
        team1_name = matchup_data.loc[i, 'Team1']
        team2_name = matchup_data.loc[i, 'Team2']
        try:
            teamID1 = getTeamID(team1_name)
        except:
            b = True
            logger.error(
                F'{team1_name} is spelled incorrectly. Please go to the MTeams.csv file to find the correct spelling.')

        try:
            teamID2 = getTeamID(team2_name)
        except:
            b = True
            logger.error(
                F'{team2_name} is spelled incorrectly. Please go to the MTeams.csv file to find the correct spelling.')

        if b:
            continue
        else:
            t1_data = getSeasonData(teamID1, 2023)
            t2_data = getSeasonData(teamID2, 2023)
            nums.append(getOurNums(t1_data, t2_data))

    if b:
        logger.error('Make sure there are no leading or trailing spaces. i.e. This is correct: A,B while this is not: __A,__B__ where the underscores represent the spaces.')
        sys.exit()
    else:
        headers = ['Wins', 'PPG', 'OPPG', 'P6', '3PT', 'TO', 'AST',
                   'ConfChamp', 'Seed', 'RPG', 'STL', 'NumApp', 'Home']
        nums_df = pd.DataFrame(nums, columns=headers)
        nums_df.to_csv('matchup_results.csv', index=False)

for i in range(len(matchup_data) - 1, -1, -1):
    team1_name = getTeamName(matchup_data[i][0])
    team2_name = getTeamName(matchup_data[i][1])
    print(F'Matchup {i + 1}:')
    print(F'Team 1: {team1_name}')
    print(F'Team 2: {team2_name} \n')
print("Remeber, the LAST matchup you put on your file is the FIRST thing in the Predictions Data (i.e. the Top)")
