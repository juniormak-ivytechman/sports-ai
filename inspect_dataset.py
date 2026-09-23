import pandas as pd


file_path = "data/premier_league_features.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


print("Total matches:", len(df))
print()

print("First match:")
print(df["Date"].min())

print()

print("Last match:")
print(df["Date"].max())

print()

print("Matches by year:")
print(
    df["Date"]
    .dt.year
    .value_counts()
    .sort_index()
)