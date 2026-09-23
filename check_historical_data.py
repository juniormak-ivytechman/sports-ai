from pathlib import Path
import pandas as pd


data_folder = Path("data")

files = list(data_folder.glob("premier_league_*.csv"))

total_matches = 0

print("Files found:")
print()

for file in files:
    df = pd.read_csv(file)

    print(file.name, "->", len(df), "matches")

    total_matches += len(df)

print()
print("Total matches:", total_matches)