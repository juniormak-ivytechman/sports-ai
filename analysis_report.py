import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

file_path = "data/premier_league_features.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# DIFFERENCE FEATURES
# ---------------------------------------

df["FormPointsDifference"] = (
    df["HomeFormPoints"]
    - df["AwayFormPoints"]
)

df["HomeAwayFormDifference"] = (
    df["HomeTeamHomeFormPoints"]
    - df["AwayTeamAwayFormPoints"]
)

df["GoalsScoredDifference"] = (
    df["HomeAvgGoalsScored"]
    - df["AwayAvgGoalsScored"]
)

df["GoalsConcededDifference"] = (
    df["HomeAvgGoalsConceded"]
    - df["AwayAvgGoalsConceded"]
)

df["ShotsDifference"] = (
    df["HomeAvgShots"]
    - df["AwayAvgShots"]
)

df["ShotsOnTargetDifference"] = (
    df["HomeAvgShotsOnTarget"]
    - df["AwayAvgShotsOnTarget"]
)

df["CornersDifference"] = (
    df["HomeAvgCorners"]
    - df["AwayAvgCorners"]
)


# ---------------------------------------
# REMOVE MATCHES WITHOUT HISTORY
# ---------------------------------------

df = df[
    (df["HomeFormPoints"] > 0) |
    (df["AwayFormPoints"] > 0)
].copy()


# ---------------------------------------
# FEATURES
# ---------------------------------------

features = [
    "HomeFormPoints",
    "AwayFormPoints",

    "HomeTeamHomeFormPoints",
    "AwayTeamAwayFormPoints",

    "HomeAvgGoalsScored",
    "AwayAvgGoalsScored",

    "HomeAvgGoalsConceded",
    "AwayAvgGoalsConceded",

    "HomeAvgShots",
    "AwayAvgShots",

    "HomeAvgShotsOnTarget",
    "AwayAvgShotsOnTarget",

    "HomeAvgCorners",
    "AwayAvgCorners",

    "FormPointsDifference",
    "HomeAwayFormDifference",

    "GoalsScoredDifference",
    "GoalsConcededDifference",

    "ShotsDifference",
    "ShotsOnTargetDifference",

    "CornersDifference"
]


# ---------------------------------------
# TRAIN / TEST
# ---------------------------------------

train_df = df[df["Date"] < "2025-08-01"].copy()

test_df = df[df["Date"] >= "2025-08-01"].copy()


X_train = train_df[features]
y_train = train_df["FTR"]

X_test = test_df[features]
y_test = test_df["FTR"]


# ---------------------------------------
# MODEL
# ---------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ---------------------------------------
# TRAIN
# ---------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------
# MODEL PROBABILITIES
# ---------------------------------------

probabilities = model.predict_proba(X_test)

classes = model.named_steps["classifier"].classes_


probability_lookup = {
    class_name: index
    for index, class_name
    in enumerate(classes)
}


test_results = test_df[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "AvgH",
        "AvgD",
        "AvgA"
    ]
].reset_index(drop=True)


test_results["ModelHome"] = probabilities[
    :,
    probability_lookup["H"]
]

test_results["ModelDraw"] = probabilities[
    :,
    probability_lookup["D"]
]

test_results["ModelAway"] = probabilities[
    :,
    probability_lookup["A"]
]


# ---------------------------------------
# MARKET PROBABILITIES
# ---------------------------------------

test_results["RawMarketHome"] = (
    1 / test_results["AvgH"]
)

test_results["RawMarketDraw"] = (
    1 / test_results["AvgD"]
)

test_results["RawMarketAway"] = (
    1 / test_results["AvgA"]
)


market_total = (
    test_results["RawMarketHome"]
    + test_results["RawMarketDraw"]
    + test_results["RawMarketAway"]
)


test_results["MarketHome"] = (
    test_results["RawMarketHome"]
    / market_total
)

test_results["MarketDraw"] = (
    test_results["RawMarketDraw"]
    / market_total
)

test_results["MarketAway"] = (
    test_results["RawMarketAway"]
    / market_total
)


# ---------------------------------------
# MODEL VS MARKET DIFFERENCE
# ---------------------------------------

test_results["DifferenceHome"] = (
    test_results["ModelHome"]
    - test_results["MarketHome"]
)

test_results["DifferenceDraw"] = (
    test_results["ModelDraw"]
    - test_results["MarketDraw"]
)

test_results["DifferenceAway"] = (
    test_results["ModelAway"]
    - test_results["MarketAway"]
)


# ---------------------------------------
# SHOW LARGEST DISAGREEMENTS
# ---------------------------------------

test_results["LargestDifference"] = test_results[
    [
        "DifferenceHome",
        "DifferenceDraw",
        "DifferenceAway"
    ]
].abs().max(axis=1)


largest_disagreements = (
    test_results
    .sort_values(
        "LargestDifference",
        ascending=False
    )
    .head(20)
)


print()
print("MODEL VS MARKET ANALYSIS")
print("=" * 70)

print()

print("Largest model/market disagreements:")
print()


display_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "ModelHome",
    "ModelDraw",
    "ModelAway",
    "MarketHome",
    "MarketDraw",
    "MarketAway",
    "DifferenceHome",
    "DifferenceDraw",
    "DifferenceAway"
]


print(
    largest_disagreements[
        display_columns
    ].to_string(index=False)
)


# ---------------------------------------
# SAVE REPORT
# ---------------------------------------

output_file = (
    "data/model_vs_market_2025_26.csv"
)

test_results.to_csv(
    output_file,
    index=False
)

print()
print("Full comparison saved to:")
print(output_file)