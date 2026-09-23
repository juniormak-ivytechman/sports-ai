import pandas as pd
import numpy as np


# ---------------------------------------
# SETTINGS
# ---------------------------------------

DATA_FILE = "data/premier_league_master.csv"

STARTING_RATING = 1500

K_FACTOR = 40
HOME_ADVANTAGE = 40

# Width of the Elo-difference neighbourhood.
# Smaller = more local.
# Larger = smoother.
BAND_WIDTH = 75

# Minimum number of historical matches we
# want in a neighbourhood before trusting it.
MIN_MATCHES = 50


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d"
)

df = df.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)


# ---------------------------------------
# BUILD PRE-MATCH ELO
# ---------------------------------------

ratings = {}

elo_rows = []


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


for _, match in df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_rating = get_rating(home_team)
    away_rating = get_rating(away_team)

    difference = (
        home_rating
        + HOME_ADVANTAGE
        - away_rating
    )

    expected_home = expected_home_probability(
        home_rating,
        away_rating
    )


    elo_rows.append({
        "Date": match["Date"],
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTR": match["FTR"],
        "HomeElo": home_rating,
        "AwayElo": away_rating,
        "EloDifference": difference,
        "EloHomeWinBinaryProbability": expected_home
    })


    # -----------------------------------
    # UPDATE AFTER THE MATCH
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


elo_df = pd.DataFrame(elo_rows)


# ---------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------

calibration_df = elo_df[
    elo_df["Date"] < "2025-08-01"
].copy()

test_df = elo_df[
    elo_df["Date"] >= "2025-08-01"
].copy()


# ---------------------------------------
# CONVERT ELO DIFFERENCE INTO 1X2
# ---------------------------------------

def estimate_1x2_probability(
    difference,
    historical_df
):

    # Find historical matches with a similar
    # Elo difference.

    local = historical_df[
        (
            historical_df["EloDifference"]
            >= difference - BAND_WIDTH
        )
        &
        (
            historical_df["EloDifference"]
            <= difference + BAND_WIDTH
        )
    ]


    # If too few matches are found, gradually
    # widen the band.

    current_band = BAND_WIDTH

    while (
        len(local) < MIN_MATCHES
        and current_band < 400
    ):

        current_band += 25

        local = historical_df[
            (
                historical_df["EloDifference"]
                >= difference - current_band
            )
            &
            (
                historical_df["EloDifference"]
                <= difference + current_band
            )
        ]


    # Laplace smoothing.
    #
    # This prevents any one neighbourhood from
    # producing an extreme 0% or 100% estimate
    # simply because of a limited sample.

    home_count = (
        local["FTR"] == "H"
    ).sum()

    draw_count = (
        local["FTR"] == "D"
    ).sum()

    away_count = (
        local["FTR"] == "A"
    ).sum()


    smoothing = 1


    total = (
        home_count
        + draw_count
        + away_count
        + 3 * smoothing
    )


    home_probability = (
        home_count + smoothing
    ) / total

    draw_probability = (
        draw_count + smoothing
    ) / total

    away_probability = (
        away_count + smoothing
    ) / total


    return (
        home_probability,
        draw_probability,
        away_probability,
        len(local)
    )


# ---------------------------------------
# CALCULATE TEST PROBABILITIES
# ---------------------------------------

probabilities = []


for _, match in test_df.iterrows():

    (
        home_probability,
        draw_probability,
        away_probability,
        sample_size
    ) = estimate_1x2_probability(
        match["EloDifference"],
        calibration_df
    )


    probabilities.append({
        "HomeProbability": home_probability,
        "DrawProbability": draw_probability,
        "AwayProbability": away_probability,
        "HistoricalSampleSize": sample_size
    })


probability_df = pd.DataFrame(
    probabilities
).reset_index(drop=True)


test_df = pd.concat(
    [
        test_df.reset_index(drop=True),
        probability_df
    ],
    axis=1
)


# ---------------------------------------
# VERIFY PROBABILITIES
# ---------------------------------------

test_df["ProbabilityTotal"] = (
    test_df["HomeProbability"]
    + test_df["DrawProbability"]
    + test_df["AwayProbability"]
)


# ---------------------------------------
# PREDICTION
# ---------------------------------------

test_df["Prediction"] = np.select(
    [
        (
            test_df["HomeProbability"]
            >= test_df["DrawProbability"]
        )
        &
        (
            test_df["HomeProbability"]
            >= test_df["AwayProbability"]
        ),

        (
            test_df["DrawProbability"]
            >= test_df["HomeProbability"]
        )
        &
        (
            test_df["DrawProbability"]
            >= test_df["AwayProbability"]
        )
    ],
    [
        "H",
        "D"
    ],
    default="A"
)


# ---------------------------------------
# ACCURACY
# ---------------------------------------

accuracy = (
    test_df["Prediction"]
    == test_df["FTR"]
).mean()


# ---------------------------------------
# LOG LOSS
# ---------------------------------------

from sklearn.metrics import log_loss


probability_matrix = test_df[
    [
        "AwayProbability",
        "DrawProbability",
        "HomeProbability"
    ]
].values


model_log_loss = log_loss(
    test_df["FTR"],
    probability_matrix,
    labels=["A", "D", "H"]
)


# ---------------------------------------
# OUTPUT
# ---------------------------------------

print()
print("ELO 1X2 MODEL")
print("=" * 70)

print()

print(
    "Calibration period:",
    calibration_df["Date"].min().date(),
    "to",
    calibration_df["Date"].max().date()
)

print()

print(
    "Test period:",
    test_df["Date"].min().date(),
    "to",
    test_df["Date"].max().date()
)

print()

print("K factor:", K_FACTOR)
print("Home advantage:", HOME_ADVANTAGE)
print("Band width:", BAND_WIDTH)

print()

print(
    f"Accuracy: {accuracy:.2%}"
)

print(
    f"Log loss: {model_log_loss:.4f}"
)

print()

print("Predictions:")

print(
    test_df["Prediction"]
    .value_counts()
    .sort_index()
)

print()

print("Actual results:")

print(
    test_df["FTR"]
    .value_counts()
    .sort_index()
)

print()

print(
    "Average historical sample size:",
    f"{test_df['HistoricalSampleSize'].mean():.1f}"
)

print()

print(
    "Probability total range:"
)

print(
    test_df["ProbabilityTotal"].min(),
    "to",
    test_df["ProbabilityTotal"].max()
)

print()

print("First 15 predictions:")

print()

print(
    test_df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "HomeProbability",
            "DrawProbability",
            "AwayProbability",
            "Prediction",
            "HistoricalSampleSize"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ---------------------------------------
# SAVE
# ---------------------------------------

output_file = (
    "data/elo_1x2_2025_26.csv"
)

test_df.to_csv(
    output_file,
    index=False
)

print()

print(
    "Saved to:",
    output_file
)