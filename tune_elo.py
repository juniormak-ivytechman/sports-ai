import pandas as pd


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

file_path = "data/premier_league_elo.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# SPLIT DATA
# ---------------------------------------

# We tune using everything BEFORE 2025/26.
# 2025/26 remains completely unseen.

tune_df = df[
    df["Date"] < "2025-08-01"
].copy()


# ---------------------------------------
# PARAMETER SEARCH
# ---------------------------------------

k_values = [
    10,
    15,
    20,
    25,
    30,
    35,
    40
]

home_advantage_values = [
    40,
    50,
    60,
    70,
    80,
    90
]


STARTING_RATING = 1500


# ---------------------------------------
# EVALUATE ONE PARAMETER SET
# ---------------------------------------

def evaluate_elo(
    matches,
    k_factor,
    home_advantage
):

    ratings = {}

    correct = 0
    total = 0


    for _, match in matches.iterrows():

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_rating = ratings.get(
            home_team,
            STARTING_RATING
        )

        away_rating = ratings.get(
            away_team,
            STARTING_RATING
        )


        # Home advantage
        adjusted_home_rating = (
            home_rating
            + home_advantage
        )


        expected_home = (
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


        # Predict home win if probability >= 0.50
        prediction = (
            "H"
            if expected_home >= 0.50
            else "NOT_HOME"
        )


        actual = match["FTR"]

        actual_home = (
            actual == "H"
        )


        if (
            prediction == "H"
            and actual_home
        ) or (
            prediction == "NOT_HOME"
            and not actual_home
        ):

            correct += 1


        total += 1


        # Actual result for Elo update
        if actual == "H":

            actual_home_score = 1.0
            actual_away_score = 0.0

        elif actual == "D":

            actual_home_score = 0.5
            actual_away_score = 0.5

        else:

            actual_home_score = 0.0
            actual_away_score = 1.0


        # Update ratings
        ratings[home_team] = (
            home_rating
            +
            k_factor
            *
            (
                actual_home_score
                - expected_home
            )
        )


        ratings[away_team] = (
            away_rating
            +
            k_factor
            *
            (
                actual_away_score
                - (1 - expected_home)
            )
        )


    accuracy = correct / total

    return accuracy


# ---------------------------------------
# SEARCH
# ---------------------------------------

results = []


for k_factor in k_values:

    for home_advantage in home_advantage_values:

        accuracy = evaluate_elo(
            tune_df,
            k_factor,
            home_advantage
        )


        results.append({
            "KFactor": k_factor,
            "HomeAdvantage": home_advantage,
            "Accuracy": accuracy
        })


results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)


# ---------------------------------------
# OUTPUT
# ---------------------------------------

print()

print("ELO PARAMETER SEARCH")
print("=" * 60)

print()

print("Best parameter combinations:")
print()

print(
    results_df
    .head(10)
    .to_string(index=False)
)


print()

best = results_df.iloc[0]

print("BEST PARAMETERS")
print("-" * 30)

print(
    "K factor:",
    int(best["KFactor"])
)

print(
    "Home advantage:",
    int(best["HomeAdvantage"])
)

print(
    f"Validation accuracy: "
    f"{best['Accuracy']:.2%}"
)