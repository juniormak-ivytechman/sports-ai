import pandas as pd


file_path = "data/premier_league_features.csv"

df = pd.read_csv(file_path)

# Find matches where the home team had already earned
# at least 9 points from its previous 5 home matches.
positive_home_form = df[
    df["HomeTeamHomeFormPoints"] >= 9
]

print("Examples of strong home form:")
print()

print(
    positive_home_form[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeTeamHomeFormPoints"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()

# Find matches where the away team had already earned
# at least 9 points from its previous 5 away matches.
positive_away_form = df[
    df["AwayTeamAwayFormPoints"] >= 9
]

print("Examples of strong away form:")
print()

print(
    positive_away_form[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "AwayTeamAwayFormPoints"
        ]
    ]
    .head(10)
    .to_string(index=False)
)