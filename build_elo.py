import duckdb
import pandas as pd


DATABASE_PATH = "database/sports_ai.duckdb"

STARTING_RATING = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 65


# ---------------------------------------
# LOAD MATCHES
# ---------------------------------------

connection = duckdb.connect(DATABASE_PATH)

df = connection.execute("""
    SELECT
        Date,
        HomeTeam,
        AwayTeam,
        FTR
    FROM premier_league_matches
    ORDER BY Date
""").fetchdf()

connection.close()

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# TEAM RATINGS
# ---------------------------------------

ratings = {}


def get_rating(team):
    """
    Return the team's current Elo rating.

    New teams start at 1500.
    """
    return ratings.get(team, STARTING_RATING)


def expected_home_probability(
    home_rating,
    away_rating
):
    """
    Calculate expected home win probability.

    Home advantage is added to the home team's rating.
    """

    adjusted_home_rating = (
        home_rating + HOME_ADVANTAGE
    )

    probability = (
        1
        /
        (
            1
            +
            10
            ** (
                (
                    away_rating
                    - adjusted_home_rating
                )
                / 400
            )
        )
    )

    return probability


# ---------------------------------------
# FEATURE STORAGE
# ---------------------------------------

home_elo_before = []
away_elo_before = []
elo_difference = []

home_expected_probability = []


# ---------------------------------------
# PROCESS MATCHES
# ---------------------------------------

for _, match in df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_rating = get_rating(home_team)
    away_rating = get_rating(away_team)


    # Store ratings BEFORE the match.
    #
    # This is extremely important.
    # The current match must not influence
    # the feature used to predict itself.

    home_elo_before.append(home_rating)
    away_elo_before.append(away_rating)

    elo_difference.append(
        (
            home_rating
            + HOME_ADVANTAGE
            - away_rating
        )
    )


    probability = expected_home_probability(
        home_rating,
        away_rating
    )

    home_expected_probability.append(
        probability
    )


    # -----------------------------------
    # ACTUAL RESULT
    # -----------------------------------

    if match["FTR"] == "H":
        actual_home = 1.0
        actual_away = 0.0

    elif match["FTR"] == "D":
        actual_home = 0.5
        actual_away = 0.5

    else:
        actual_home = 0.0
        actual_away = 1.0


    # -----------------------------------
    # EXPECTED RESULT
    # -----------------------------------

    expected_home = probability
    expected_away = 1 - probability


    # -----------------------------------
    # UPDATE RATINGS
    # -----------------------------------

    new_home_rating = (
        home_rating
        +
        K_FACTOR
        *
        (
            actual_home
            -
            expected_home
        )
    )

    new_away_rating = (
        away_rating
        +
        K_FACTOR
        *
        (
            actual_away
            -
            expected_away
        )
    )


    ratings[home_team] = new_home_rating
    ratings[away_team] = new_away_rating


# ---------------------------------------
# ADD FEATURES
# ---------------------------------------

df["HomeElo"] = home_elo_before
df["AwayElo"] = away_elo_before
df["EloDifference"] = elo_difference
df["EloHomeProbability"] = home_expected_probability


# ---------------------------------------
# SAVE
# ---------------------------------------

output_file = "data/premier_league_elo.csv"

df.to_csv(
    output_file,
    index=False
)


print()
print("Elo dataset created.")
print("Total matches:", len(df))
print("Saved to:", output_file)

print()

print(
    df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeElo",
            "AwayElo",
            "EloDifference",
            "EloHomeProbability",
            "FTR"
        ]
    ]
    .tail(20)
    .to_string(index=False)
)


print()

print("Current Elo ratings:")

current_ratings = (
    pd.DataFrame(
        [
            {
                "Team": team,
                "Elo": rating
            }
            for team, rating
            in ratings.items()
        ]
    )
    .sort_values(
        "Elo",
        ascending=False
    )
)

print(
    current_ratings
    .head(20)
    .to_string(index=False)
)