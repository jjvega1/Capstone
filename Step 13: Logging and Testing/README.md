There are two ways to run this code; you can either pass one or three arguments. If you do not give one or three arguments, there will be an error, so make sure you avoid that.

If you want to view the results of a given year, you would pass in one argument containing the year. If we want to see how the model does in a given year say 2005, we type the following:

python3 production_ready.py 2005 (For Mac Users)
python production_ready.py 2005 (For windows users)

Make sure to put in valid years. Valid years are any year from 1985 - 2022 inclusive on both, and the year cannot be 2020 since there was no tournament that year.

If you want to look at a certain matchup between two teams in a given year, you will pass in three arguments. The first one is the year (make sure it is valid). The second and third arguments are the Team's ID. To look for a Team's ID, you can go to the Teams.csv file in the Data folder and Ctrl+f (or Cmd+f for Mac Users) your team. Your team might not show up, and if it does not, slightly change the spelling of your team. For example, if you want to look up FDU and it does not show up, try F Dickinsion, and you will find it. This is in mostly alphabetical order with a few exceptions towards the bottom, so you should find your Team's ID really quickly. You can also make a call to the getTeamID() function and pass in the appropiately named college. Here is an example of making the call with three arguments:

python production_ready.py 2004 1181 1246 (To look at a matchup between Duke and Kentucky in 2004)

The result will be the probability that the first Team ID's wins. Note that you can choose any two teams you want; it does not have to be tournament participants. Do be careful however; make sure the Team ID's are Division I at the given year. If not, you will have some very shaky results, but that really should not matter since teams that have been in and out of Division I are most likely terrible compared to the top. 

Also do not forget that valid Team ID's are from 1401 - 1477 respectively.