import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/results.csv")

# Quick look
print(f"Total matches: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMatch types (top 10):")
print(df['tournament'].value_counts().head(10))


df = pd.read_csv("data/raw/results.csv")
df['date'] = pd.to_datetime(df['date'])

# Most recent matches
print("\nMost recent 10 matches:")
print(df.sort_values('date', ascending=False).head(10)[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']])

# Match counts by year (last 10 years)
print("\nMatches per year (last 10 years):")
print(df[df['date'] >= '2016-01-01'].groupby(df['date'].dt.year).size())



df = pd.read_csv("data/raw/results.csv")
df['date'] = pd.to_datetime(df['date'])

# Filter to matches that have actual scores
played = df.dropna(subset=['home_score', 'away_score'])

print(f"Total matches: {len(df)}")
print(f"Played matches (have scores): {len(played)}")
print(f"Future fixtures (no scores): {len(df) - len(played)}")
print(f"\nMost recent 5 PLAYED matches:")
print(played.sort_values('date', ascending=False).head(5)[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']])