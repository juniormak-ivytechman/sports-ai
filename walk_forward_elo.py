import pandas as pd


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

file_path = "data/premier_league_master.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d"
)


# ---------------------------------------
# SETTINGS
# ---------------------------------------

STARTING_RATING = 1500

K_FACTOR = 40
HOME_ADVANTAGE = 40


# ---------------------------------------
# ELO FUNCTION
# ---------------------------------------

def run_elo(train_df, test_df):

    ratings = {}

    correct = 0
    total = 0

    probabilities = []
    actuals = []


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


    # -----------------------------------
    # TRAIN / WARM UP
    # -----------------------------------

    for _, match in train_df.iterrows():

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_rating = get_rating(home_team)
        away_rating = get_rating(away_team)

        probability = expected_home_probability(
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


    # -----------------------------------
    # TEST
    # -----------------------------------

    for _, match in test_df.iterrows():

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_rating = get_rating(home_team)
        away_rating = get_rating(away_team)


        probability = expected_home_probability(
            home_rating,
            away_rating
        )


        prediction = (
            1
            if probability >= 0.50
            else 0
        )


        actual = (
            1
            if match["FTR"] == "H"
            else 0
        )


        if prediction == actual:
            correct += 1

        total += 1

        probabilities.append(probability)
        actuals.append(actual)


        # --------------------------------
        # UPDATE AFTER PREDICTION
        # --------------------------------

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


    accuracy = correct / total

    return accuracy


# ---------------------------------------
# WALK-FORWARD WINDOWS
# ---------------------------------------

windows = [

    {
        "name": "2022/23",
        "train_end": "2022-08-01",
        "test_end": "2023-06-01"
    },

    {
        "name": "2023/24",
        "train_end": "2023-08-01",
        "test_end": "2024-06-01"
    },

    {
        "name": "2024/25",
        "train_end": "2024-08-01",
        "test_end": "2025-06-01"
    },

    {
        "name": "2025/26 FINAL",
        "train_end": "2025-08-01",
        "test_end": "2026-06-01"
    }
]


# ---------------------------------------
# RUN WINDOWS
# ---------------------------------------

results = []


for window in windows:

    train_df = df[
        df["Date"] < window["train_end"]
    ].copy()


    test_df = df[
        (
            df["Date"] >= window["train_end"]
        )
        &
        (
            df["Date"] < window["test_end"]
        )
    ].copy()


    accuracy = run_elo(
        train_df,
        test_df
    )


    results.append({
        "TestPeriod": window["name"],
        "TrainMatches": len(train_df),
        "TestMatches": len(test_df),
        "Accuracy": accuracy
    })


# ---------------------------------------
# OUTPUT
# ---------------------------------------

results_df = pd.DataFrame(results)

print()
print("WALK-FORWARD ELO EVALUATION")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2%}".format
        }
    )
)