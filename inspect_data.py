import pandas as pd


file_path = "data/premier_league_2024_25.csv"

df = pd.read_csv(file_path)

print("Number of matches:", len(df))
print()

print("Columns:")
print(df.columns.tolist())
print()

print("First 5 matches:")
print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].head())