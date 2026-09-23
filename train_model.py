import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

file_path = "data/premier_league_features.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# CREATE DIFFERENCE FEATURES
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
# REMOVE MATCHES WITH NO PREVIOUS HISTORY
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
# TRAIN / TEST BY SEASON
# ---------------------------------------

# 2025/26 is our completely unseen test season.

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
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# ---------------------------------------
# TRAIN
# ---------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------
# CLASS PREDICTIONS
# ---------------------------------------

predictions = model.predict(X_test)


# ---------------------------------------
# PROBABILITY PREDICTIONS
# ---------------------------------------

probabilities = model.predict_proba(X_test)

class_order = model.named_steps["classifier"].classes_

probability_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Probability_{class_name}"
        for class_name in class_order
    ]
)


# ---------------------------------------
# METRICS
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

logloss = log_loss(
    y_test,
    probabilities,
    labels=class_order
)


# ---------------------------------------
# OUTPUT
# ---------------------------------------

print()
print("MODEL #3")
print("=" * 60)

print()

print("Training matches:", len(train_df))
print("Testing matches:", len(test_df))

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

print(f"Accuracy: {accuracy:.2%}")
print(f"Log loss: {logloss:.4f}")

print()

print("Predictions:")
print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)

print()

print("Actual results:")
print(
    y_test
    .value_counts()
    .sort_index()
)

print()

print("Confusion matrix:")
print()

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["A", "D", "H"]
)

print(
    pd.DataFrame(
        matrix,
        index=["Actual A", "Actual D", "Actual H"],
        columns=["Pred A", "Pred D", "Pred H"]
    )
)

print()

print("Classification report:")
print()

print(
    classification_report(
        y_test,
        predictions,
        labels=["A", "D", "H"],
        zero_division=0
    )
)


# ---------------------------------------
# SHOW FIRST 10 PROBABILITY PREDICTIONS
# ---------------------------------------

results = test_df[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTR"
    ]
].reset_index(drop=True)

results = pd.concat(
    [
        results,
        probability_df.reset_index(drop=True)
    ],
    axis=1
)

print()

print("First 10 probability predictions:")
print()

print(
    results.head(10).to_string(index=False)
)