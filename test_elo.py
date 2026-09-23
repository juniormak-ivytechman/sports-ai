import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss
)


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

file_path = "data/premier_league_elo.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# PARAMETERS SELECTED FROM VALIDATION
# ---------------------------------------

STARTING_RATING = 1500

K_FACTOR = 40
HOME_ADVANTAGE = 40


# ---------------------------------------
# SPLIT DATA
# ---------------------------------------

train_df = df[
    df["Date"] < "2025-08-01"
].copy()

test_df = df[
    df["Date"] >= "2025-08-01"
].copy()


# ---------------------------------------
# BUILD ELO FROM TRAINING PERIOD
# ---------------------------------------

ratings = {}


def get_rating(team):
    return ratings.get(
        team,
        STARTING_RATING
    )


def expected_home_probability(
    home_rating,
    away_rating
):

    adjusted_home_rating = (
        home_rating
        + HOME_ADVANTAGE
    )

    return (
        1
        /
        (
            1
            +
            10
            **
            (
                (
                    away_rating
                    - adjusted_home_rating
                )
                / 400
            )
        )
    )


# Process every training match
# to establish the ratings entering
# the 2025/26 season.

for _, match in train_df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_rating = get_rating(home_team)
    away_rating = get_rating(away_team)

    expected_home = expected_home_probability(
        home_rating,
        away_rating
    )


    if match["FTR"] == "H":

        actual_home = 1.0
        actual_away = 0.0

    elif match["FTR"] == "D":

        actual_home = 0.5
        actual_away = 0.5

    else:

        actual_home = 0.0
        actual_away = 1.0


    ratings[home_team] = (
        home_rating
        +
        K_FACTOR
        *
        (
            actual_home
            - expected_home
        )
    )

    ratings[away_team] = (
        away_rating
        +
        K_FACTOR
        *
        (
            actual_away
            - (1 - expected_home)
        )
    )


# ---------------------------------------
# TEST 2025/26
# ---------------------------------------

predicted_home_probabilities = []

actual_home_results = []


for _, match in test_df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_rating = get_rating(home_team)
    away_rating = get_rating(away_team)


    probability = expected_home_probability(
        home_rating,
        away_rating
    )


    predicted_home_probabilities.append(
        probability
    )

    actual_home_results.append(
        1
        if match["FTR"] == "H"
        else 0
    )


    # Update Elo AFTER prediction.
    #
    # The match result must not influence
    # the prediction for the same match.

    if match["FTR"] == "H":

        actual_home = 1.0
        actual_away = 0.0

    elif match["FTR"] == "D":

        actual_home = 0.5
        actual_away = 0.5

    else:

        actual_home = 0.0
        actual_away = 1.0


    ratings[home_team] = (
        home_rating
        +
        K_FACTOR
        *
        (
            actual_home
            - probability
        )
    )

    ratings[away_team] = (
        away_rating
        +
        K_FACTOR
        *
        (
            actual_away
            - (1 - probability)
        )
    )


# ---------------------------------------
# CONVERT PROBABILITIES TO PREDICTIONS
# ---------------------------------------

predictions = [
    1 if probability >= 0.50 else 0
    for probability
    in predicted_home_probabilities
]


# ---------------------------------------
# METRICS
# ---------------------------------------

accuracy = accuracy_score(
    actual_home_results,
    predictions
)

home_log_loss = log_loss(
    actual_home_results,
    predicted_home_probabilities
)


# ---------------------------------------
# OUTPUT
# ---------------------------------------

print()
print("FINAL OUT-OF-SAMPLE ELO TEST")
print("=" * 60)

print()

print("Training period:")
print(
    train_df["Date"].min().date(),
    "to",
    train_df["Date"].max().date()
)

print()

print("Testing period:")
print(
    test_df["Date"].min().date(),
    "to",
    test_df["Date"].max().date()
)

print()

print("K factor:", K_FACTOR)
print("Home advantage:", HOME_ADVANTAGE)

print()

print(
    f"Home-win accuracy: {accuracy:.2%}"
)

print(
    f"Home-win log loss: {home_log_loss:.4f}"
)

print()

print(
    "Actual home wins:",
    sum(actual_home_results)
)

print(
    "Predicted home wins:",
    sum(predictions)
)