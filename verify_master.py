import pandas as pd


file_path = "data/premier_league_master.csv"

df = pd.read_csv(file_path)

print("Total matches:", len(df))
print()

print("Matches by season:")
print(df["Season"].value_counts().sort_index())
print()

print("Unique matches:", df[["Date", "HomeTeam", "AwayTeam", "Season"]].drop_duplicates().shape[0])
print()

print("Missing values in important columns:")

important_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR"
]

print(df[important_columns].isna().sum())