import pandas as pd
import numpy as np

# Load data
hof = pd.read_csv('data/HallOfFame.csv')
batting = pd.read_csv('data/Batting.csv')
people = pd.read_csv('data/People.csv')

# Keep only players (not managers/umpires) and one row per player
# Use max inducted status — if ever inducted, mark as 1
hof_players = hof[hof['category'] == 'Player'].copy()
hof_inducted = hof_players.groupby('playerID')['inducted'].apply(
    lambda x: 1 if 'Y' in x.values else 0
).reset_index()
hof_inducted.columns = ['playerID', 'inducted']

# Aggregate batting stats by player career totals
career = batting.groupby('playerID').agg(
    G=('G', 'sum'),
    AB=('AB', 'sum'),
    R=('R', 'sum'),
    H=('H', 'sum'),
    HR=('HR', 'sum'),
    RBI=('RBI', 'sum'),
    SB=('SB', 'sum'),
    BB=('BB', 'sum'),
    SO=('SO', 'sum'),
    seasons=('yearID', 'nunique')
).reset_index()

# Calculate rate stats
career['BA'] = career['H'] / career['AB']
career['OBP'] = (career['H'] + career['BB']) / (career['AB'] + career['BB'])
career['HR_rate'] = career['HR'] / career['AB']
career['K_rate'] = career['SO'] / career['AB']

# Merge with HOF data
df = career.merge(hof_inducted, on='playerID', how='inner')

# Drop players with less than 1000 AB (pitchers, short careers)
df = df[df['AB'] >= 1000].copy()

# Fill any NaN
df = df.fillna(0)

print("Dataset shape:", df.shape)
print("HOF inducted:", df['inducted'].sum())
print("Not inducted:", (df['inducted'] == 0).sum())
print("\nSample:")
print(df.head())

# Save clean dataset
df.to_csv('data/model_data.csv', index=False)
print("\nSaved to data/model_data.csv")