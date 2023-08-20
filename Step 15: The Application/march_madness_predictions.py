import pandas as pd
import warnings
warnings.filterwarnings('ignore')

teams_pd = pd.read_csv('Data/MTeams.csv')
seeds_pd = pd.read_csv('Data/MNCAATourneySeeds.csv')
reg_season_compact_pd = pd.read_csv('Data/MRegularSeasonCompactResults.csv')
reg_season_detailed_pd = pd.read_csv('Data/MRegularSeasonDetailedResults.csv')
teams_pd = pd.read_csv('Data/MTeams.csv')
tourney_compact_pd = pd.read_csv('Data/MNCAATourneyCompactResults.csv')
conference_tourney_results_pd = pd.read_csv('Data/MConferenceTourneyGames.csv')
conferences_pd = pd.read_csv('Data/MTeamConferences.csv')

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


playin = [1394, 1338, 1192, 1113]

# Get the first round matchups
matchups = []

# Helper function to swap


def swap_it(m, x1, x2):
    m[x1], m[x2] = m[x2], m[x1]
    return m


k = 0

# We will have to find the playin winner from each quadrant
for i in range(8):
    seeds_pd_2023 = seeds_pd[seeds_pd['Season'] == 2023]
    if len(seeds_pd_2023.iloc[16 - i, 1]) == 4:
        if seeds_pd_2023.iloc[16 - i, 2] not in playin:
            k = 1
    matchups.append(
        [seeds_pd_2023.iloc[i, 2], seeds_pd_2023.iloc[16 - i - k, 2]])

k = 0

for i in range(17, 25):
    if len(seeds_pd_2023.iloc[50 - i, 1]) == 4:
        if seeds_pd_2023.iloc[50 - i, 2] not in playin:
            k = 1
    matchups.append(
        [seeds_pd_2023.iloc[i, 2], seeds_pd_2023.iloc[50 - i - k, 2]])

k = 0

for i in range(34, 42):
    if len(seeds_pd_2023.iloc[84 - i, 1]) == 4:
        if seeds_pd_2023.iloc[84 - i, 2] not in playin:
            k = 1
    matchups.append(
        [seeds_pd_2023.iloc[i, 2], seeds_pd_2023.iloc[84 - i - k, 2]])

k = 0

for i in range(51, 59):
    if len(seeds_pd_2023.iloc[118 - i, 1]) == 4:
        if seeds_pd_2023.iloc[118 - i, 2] not in playin:
            k = 1
    matchups.append(
        [seeds_pd_2023.iloc[i, 2], seeds_pd_2023.iloc[118 - i - k, 2]])

for i in range(0, 4):
    matchups = swap_it(matchups, 8 * i + 1, 8 * i + 7)
    matchups = swap_it(matchups, 8 * i + 2, 8 * i + 4)

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


def predictOutcome(team_data1, team_data2):
    diff = [a - b for a, b in zip(team_data1, team_data2)]
    return diff


print("Round 1 START! \n")
list_of_vectors = []

# Get the first round results
for i in range(len(matchups) - 1, -1, -1):
    team1_name = getTeamName(matchups[i][0])
    team2_name = getTeamName(matchups[i][1])

    team1_data = getSeasonData(matchups[i][0], 2023)
    team2_data = getSeasonData(matchups[i][1], 2023)
    vec = predictOutcome(team1_data, team2_data)
    list_of_vectors.append(vec)

headers = ['Wins', 'PPG', 'OPPG', 'P6', '3PT', 'TO', 'AST',
                   'ConfChamp', 'Seed', 'RPG', 'STL', 'NumApp', 'Home']
list_of_vec_df = pd.DataFrame(list_of_vectors, columns=headers)
list_of_vec_df.to_csv('mm_2023_r1.csv', index=False)

r1_winners = []

for i in range(len(matchups) - 1, -1, -1):
    team1_name = getTeamName(matchups[i][0])
    team2_name = getTeamName(matchups[i][1])
    print(F'Matchup {i + 1}:')
    print(F'Team 1: {team1_name}')
    print(F'Team 2: {team2_name}')
    if i == 1 or i == 6 or i == 10 or i == 21 or i == 25 or i == 26 or i == 29:
        print(F'{team2_name} wins \n')
        r1_winners.append(team2_name)
    else:
        print(F'{team1_name} wins \n')
        r1_winners.append(team1_name)

print("Round 2 START! \n")

r2_matchups = []

for i in range(len(r1_winners) - 1, 0, -2):
    r2_matchups.append([r1_winners[i], r1_winners[i-1]])

list_of_vectors = []

for i in range(len(r2_matchups) - 1, -1, -1):
    team1_ID = getTeamID(r2_matchups[i][0])
    team2_ID = getTeamID(r2_matchups[i][1])

    team1_data = getSeasonData(team1_ID, 2023)
    team2_data = getSeasonData(team2_ID, 2023)
    vec = predictOutcome(team1_data, team2_data)
    list_of_vectors.append(vec)

list_of_vec_df = pd.DataFrame(list_of_vectors, columns=headers)
list_of_vec_df.to_csv('mm_2023_r2.csv', index=False)

r2_winners = []

for i in range(len(r2_matchups) - 1, -1, -1):
    team1_name = r2_matchups[i][0]
    team2_name = r2_matchups[i][1]
    print(F'Matchup {i + 1}:')
    print(F'Team 1: {team1_name}')
    print(F'Team 2: {team2_name}')
    if i == 7 or i == 11 or i == 13 or i == 15:
        print(F'{team2_name} wins \n')
        r2_winners.append(team2_name)
    else:
        print(F'{team1_name} wins \n')
        r2_winners.append(team1_name)

print("Round 3 START! \n")

r3_matchups = []

for i in range(len(r2_winners) - 1, 0, -2):
    r3_matchups.append([r2_winners[i], r2_winners[i-1]])

list_of_vectors = []

for i in range(len(r3_matchups) - 1, -1, -1):
    team1_ID = getTeamID(r3_matchups[i][0])
    team2_ID = getTeamID(r3_matchups[i][1])

    team1_data = getSeasonData(team1_ID, 2023)
    team2_data = getSeasonData(team2_ID, 2023)
    vec = predictOutcome(team1_data, team2_data)
    list_of_vectors.append(vec)

list_of_vec_df = pd.DataFrame(list_of_vectors, columns=headers)
list_of_vec_df.to_csv('mm_2023_r3.csv', index=False)

r3_winners = []

for i in range(len(r3_matchups) - 1, -1, -1):
    team1_name = r3_matchups[i][0]
    team2_name = r3_matchups[i][1]
    print(F'Matchup {i + 1}:')
    print(F'Team 1: {team1_name}')
    print(F'Team 2: {team2_name}')
    if i == 3 or i == 5 or i == 7:
        print(F'{team2_name} wins \n')
        r3_winners.append(team2_name)
    else:
        print(F'{team1_name} wins \n')
        r3_winners.append(team1_name)

print("Round 4 START! \n")

r4_matchups = []

for i in range(len(r3_winners) - 1, 0, -2):
    r4_matchups.append([r3_winners[i], r3_winners[i-1]])

list_of_vectors = []

for i in range(len(r4_matchups) - 1, -1, -1):
    team1_ID = getTeamID(r4_matchups[i][0])
    team2_ID = getTeamID(r4_matchups[i][1])

    team1_data = getSeasonData(team1_ID, 2023)
    team2_data = getSeasonData(team2_ID, 2023)
    vec = predictOutcome(team1_data, team2_data)
    list_of_vectors.append(vec)

list_of_vec_df = pd.DataFrame(list_of_vectors, columns=headers)
list_of_vec_df.to_csv('mm_2023_r4.csv', index=False)

r4_winners = []

for i in range(len(r4_matchups) - 1, -1, -1):
    team1_name = r4_matchups[i][0]
    team2_name = r4_matchups[i][1]
    print(F'Matchup {i + 1}:')
    print(F'Team 1: {team1_name}')
    print(F'Team 2: {team2_name}')
    if i == 1 or i == 3:
        print(F'{team2_name} wins \n')
        r4_winners.append(team2_name)
    else:
        print(F'{team1_name} wins \n')
        r4_winners.append(team1_name)

print("Round 5 START! \n")

r5_matchups = []

for i in range(len(r4_winners) - 1, 0, -2):
    r5_matchups.append([r4_winners[i], r4_winners[i-1]])

list_of_vectors = []

for i in range(len(r5_matchups) - 1, -1, -1):
    team1_ID = getTeamID(r5_matchups[i][0])
    team2_ID = getTeamID(r5_matchups[i][1])

    team1_data = getSeasonData(team1_ID, 2023)
    team2_data = getSeasonData(team2_ID, 2023)
    vec = predictOutcome(team1_data, team2_data)
    list_of_vectors.append(vec)

list_of_vec_df = pd.DataFrame(list_of_vectors, columns=headers)
list_of_vec_df.to_csv('mm_2023_r5.csv', index=False)

final_matchup = []

for i in range(len(r5_matchups) - 1, -1, -1):
    team1_name = r5_matchups[i][0]
    team2_name = r5_matchups[i][1]
    print(F'Matchup {i + 1}:')
    print(F'Team 1: {team1_name}')
    print(F'Team 2: {team2_name}')
    print(F'{team2_name} wins \n')

    # Since Team 2 won on both ends
    final_matchup.append(team2_name)

print("Final Round BEGIN! \n")

team1_name = final_matchup[0]
team2_name = final_matchup[1]

team1_ID = getTeamID(team1_name)
team2_ID = getTeamID(team2_name)

team1_data = getSeasonData(team1_ID, 2023)
team2_data = getSeasonData(team2_ID, 2023)

vec = predictOutcome(team1_data, team2_data)
list_of_vec_df = pd.DataFrame([vec], columns=headers)
list_of_vec_df.to_csv('mm_2023_r6.csv', index=False)

print(F'Matchup:')
print(F'Team 1: {team1_name}')
print(F'Team 2: {team2_name}')
print(F'{team1_name} wins \n')
