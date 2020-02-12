# Set Matplotlib to print on this notebook as opposed to producing a new window
# Import libraries
import random
import matplotlib.pyplot as plt
import numpy as np

# Number of years/seasons to simulate
NumberOfYears = 100

# Teams
TeamArray = ['Arizona Diamondbacks','Atlanta Braves','Baltimore Orioles','Boston Red Sox','Chicago White Sox','Chicago Cubs',\
    'Cincinnati Reds','Cleveland Indians','Colorado Rockies','Detroit Tigers','Houston Astros',\
    'Kansas City Royals','Los Angeles Angels','Los Angeles Dodgers','Miami Marlins','Milwaukee Brewers',\
    'Minnesota Twins','New York Mets','New York Yankees','Oakland Athletics','Philadelphia Phillies','Pittsburgh Pirates',\
    'San Diego Padres','San Francisco Giants','Seattle Mariners','St. Louis Cardinals','Tampa Bay Rays','Texas Rangers',\
    'Toronto Blue Jays','Washington Nationals']
NumberOfTeams = len(TeamArray)

# Simulate the seasons
YearlyVictor = np.zeros(NumberOfYears, dtype=np.int) # Declare array for the yearly winner
for i in range(0, NumberOfYears):
    YearlyVictor[i] = random.randint(0, NumberOfTeams-1) # Array index starts at zero

# Find the number of victories for each team
NumberOfVictories = np.zeros(NumberOfTeams, dtype=np.int)
print('Number of Victories for each team over', NumberOfYears, 'years')
NumberOfVictories = np.bincount(YearlyVictor, minlength=NumberOfTeams)
for i in range(0,NumberOfTeams):
    print(TeamArray[i] + ': ', NumberOfVictories[i])

# Find the longest drought between wins for a team
LongestLosingStreak = np.zeros(NumberOfTeams, dtype=np.int) # Initialize to zero for each team
print('Longest title drought for each team over', NumberOfYears, 'years')
for i in range(0,NumberOfTeams):
    if(NumberOfVictories[i] > 0):
        CurTeamVictorYears = np.asarray(np.where(YearlyVictor == i))
        # Append the year zero to the front of the array
        CurTeamVictorYears = np.append(0, CurTeamVictorYears)
        # Append the last year of the simulation to check the time between it and the last victory
        CurTeamVictorYears = np.append(CurTeamVictorYears, NumberOfYears-1)
        for j in range(1,len(CurTeamVictorYears)):
            CurDiff = CurTeamVictorYears[j] - CurTeamVictorYears[j-1] 
            if(CurDiff > LongestLosingStreak[i]):
                LongestLosingStreak[i] = CurDiff

    # Change the print output depending on how many each team won
    if(NumberOfVictories[i] == 0):
        print(TeamArray[i], 'did not win at all')
    else:
        print(TeamArray[i], ':', LongestLosingStreak[i])

# What is the maximum number of victories
MinVictories = min(NumberOfVictories)
MaxVictories = max(NumberOfVictories)
BinValues = np.arange(-0.5,MaxVictories+1.5,1)
plt.hist(NumberOfVictories, BinValues)
plt.xlabel('Number Of Victories')
plt.ylabel('Number Of Teams')
plt.xlim(MinVictories-1,MaxVictories+1)
plt.show()