import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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
# REMOVE FIRST MATCHES WITH NO HISTORY
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
# MODEL #4
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
# PREDICTIONS
# ---------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

classes = model.named_steps["classifier"].classes_


# ---------------------------------------
# MODEL METRICS
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

model_log_loss = log_loss(
    y_test,
    probabilities,
    labels=classes
)


# ---------------------------------------
# BOOKMAKER MARKET PROBABILITIES
# ---------------------------------------

market_df = test_df[
    [
        "AvgH",
        "AvgD",
        "AvgA"
    ]
].copy()


# Convert decimal odds into implied probabilities

market_df["RawHome"] = 1 / market_df["AvgH"]
market_df["RawDraw"] = 1 / market_df["AvgD"]
market_df["RawAway"] = 1 / market_df["AvgA"]


# Remove bookmaker margin by normalising

market_total = (
    market_df["RawHome"]
    + market_df["RawDraw"]
    + market_df["RawAway"]
)

market_df["MarketHome"] = (
    market_df["RawHome"]
    / market_total
)

market_df["MarketDraw"] = (
    market_df["RawDraw"]
    / market_total
)

market_df["MarketAway"] = (
    market_df["RawAway"]
    / market_total
)


market_probabilities = market_df[
    [
        "MarketAway",
        "MarketDraw",
        "MarketHome"
    ]
].values


# ---------------------------------------
# MARKET METRICS
# ---------------------------------------

market_log_loss = log_loss(
    y_test,
    market_probabilities,
    labels=["A", "D", "H"]
)


market_predictions = [
    ["A", "D", "H"][i]
    for i in market_probabilities.argmax(axis=1)
]

market_accuracy = accuracy_score(
    y_test,
    market_predictions
)


# ---------------------------------------
# OUTPUT
# ---------------------------------------

print()
print("MODEL #4 VS MARKET")
print("=" * 60)

print()

print("Training matches:", len(train_df))
print("Testing matches:", len(test_df))

print()

print("MODEL #4")
print("-" * 30)

print(
    f"Accuracy: {accuracy:.2%}"
)

print(
    f"Log loss: {model_log_loss:.4f}"
)

print()

print(
    "Predictions:"
)

print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)

print()

print("MARKET")
print("-" * 30)

print(
    f"Accuracy: {market_accuracy:.2%}"
)

print(
    f"Log loss: {market_log_loss:.4f}"
)

print()

print("Market implied probabilities calculated")
print("from AvgH / AvgD / AvgA after margin normalization.")