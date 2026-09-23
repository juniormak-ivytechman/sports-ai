import pandas as pd

from sklearn.metrics import log_loss


# ---------------------------------------
# LOAD ELO DATA
# ---------------------------------------

file_path = "data/premier_league_elo.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# TEST SEASON
# ---------------------------------------

test_df = df[
    df["Date"] >= "2025-08-01"
].copy()


# ---------------------------------------
# HOME WIN ACTUALS
# ---------------------------------------

actual_home_win = (
    test_df["FTR"] == "H"
).astype(int)


# ---------------------------------------
# ELO HOME WIN PROBABILITY
# ---------------------------------------

elo_home_probability = (
    test_df["EloHomeProbability"]
)


# ---------------------------------------
# ELO HOME WIN LOG LOSS
# ---------------------------------------

elo_home_log_loss = log_loss(
    actual_home_win,
    elo_home_probability
)


# ---------------------------------------
# ELO HOME-WIN CLASSIFICATION
# ---------------------------------------

elo_home_prediction = (
    elo_home_probability >= 0.50
).astype(int)


home_accuracy = (
    elo_home_prediction
    == actual_home_win
).mean()

# ---------------------------------------
# OUTPUT
# ---------------------------------------

print()
print("ELO EVALUATION")
print("=" * 50)

print()

print(
    "Testing matches:",
    len(test_df)
)

print()

print(
    f"Home-win accuracy: {home_accuracy:.2%}"
)

print(
    f"Home-win log loss: {elo_home_log_loss:.4f}"
)

print()

print("Actual home wins:")

print(
    actual_home_win.value_counts()
    .sort_index()
)