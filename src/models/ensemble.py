import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

MASTER_FILE = "data/premier_league_master.csv"
FEATURES_FILE = "data/premier_league_features.csv"
ELO_PREDICTIONS_FILE = "data/elo_1x2_tuned_2025_26.csv"
POISSON_PREDICTIONS_FILE = "data/time_decay_dixon_coles_2025_26.csv"

TEST_START = "2025-08-01"

# ============================================================
# CHECK FILES
# ============================================================

for f in [MASTER_FILE, FEATURES_FILE, ELO_PREDICTIONS_FILE, POISSON_PREDICTIONS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: Missing {f}")
        sys.exit(1)

# ============================================================
# LOAD ELO PREDICTIONS
# ============================================================

elo = pd.read_csv(ELO_PREDICTIONS_FILE)
elo["Date"] = pd.to_datetime(elo["Date"], format="%Y-%m-%d")
elo = elo[["Date", "HomeTeam", "AwayTeam", "FTR",
           "HomeProbability", "DrawProbability", "AwayProbability"]].copy()
elo = elo.rename(columns={
    "HomeProbability": "Elo_H",
    "DrawProbability": "Elo_D",
    "AwayProbability": "Elo_A",
})

# ============================================================
# LOAD POISSON + DC PREDICTIONS
# ============================================================

poisson = pd.read_csv(POISSON_PREDICTIONS_FILE)
poisson["Date"] = pd.to_datetime(poisson["Date"], format="%Y-%m-%d")
poisson = poisson[["Date", "HomeTeam", "AwayTeam",
                   "HomeProbability", "DrawProbability", "AwayProbability"]].copy()
poisson = poisson.rename(columns={
    "HomeProbability": "Poisson_H",
    "DrawProbability": "Poisson_D",
    "AwayProbability": "Poisson_A",
})

# ============================================================
# FIT ML #4 (logistic regression on engineered features)
# ============================================================

features_df = pd.read_csv(FEATURES_FILE)
features_df["Date"] = pd.to_datetime(features_df["Date"], format="%Y-%m-%d")

features_df["FormPointsDifference"] = features_df["HomeFormPoints"] - features_df["AwayFormPoints"]
features_df["HomeAwayFormDifference"] = features_df["HomeTeamHomeFormPoints"] - features_df["AwayTeamAwayFormPoints"]
features_df["GoalsScoredDifference"] = features_df["HomeAvgGoalsScored"] - features_df["AwayAvgGoalsScored"]
features_df["GoalsConcededDifference"] = features_df["HomeAvgGoalsConceded"] - features_df["AwayAvgGoalsConceded"]
features_df["ShotsDifference"] = features_df["HomeAvgShots"] - features_df["AwayAvgShots"]
features_df["ShotsOnTargetDifference"] = features_df["HomeAvgShotsOnTarget"] - features_df["AwayAvgShotsOnTarget"]
features_df["CornersDifference"] = features_df["HomeAvgCorners"] - features_df["AwayAvgCorners"]

features_df = features_df[(features_df["HomeFormPoints"] > 0) | (features_df["AwayFormPoints"] > 0)].copy()

feature_cols = [
    "HomeFormPoints", "AwayFormPoints",
    "HomeTeamHomeFormPoints", "AwayTeamAwayFormPoints",
    "HomeAvgGoalsScored", "AwayAvgGoalsScored",
    "HomeAvgGoalsConceded", "AwayAvgGoalsConceded",
    "HomeAvgShots", "AwayAvgShots",
    "HomeAvgShotsOnTarget", "AwayAvgShotsOnTarget",
    "HomeAvgCorners", "AwayAvgCorners",
    "FormPointsDifference", "HomeAwayFormDifference",
    "GoalsScoredDifference", "GoalsConcededDifference",
    "ShotsDifference", "ShotsOnTargetDifference",
    "CornersDifference",
]

train_feat = features_df[features_df["Date"] < TEST_START].copy()
test_feat = features_df[features_df["Date"] >= TEST_START].copy()

model_ml = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model_ml.fit(train_feat[feature_cols], train_feat["FTR"])

ml_probs = model_ml.predict_proba(test_feat[feature_cols])
classes = model_ml.named_steps["classifier"].classes_
class_idx = {c: i for i, c in enumerate(classes)}

ml_df = pd.DataFrame({
    "Date": test_feat["Date"].values,
    "HomeTeam": test_feat["HomeTeam"].values,
    "AwayTeam": test_feat["AwayTeam"].values,
    "ML_H": ml_probs[:, class_idx["H"]],
    "ML_D": ml_probs[:, class_idx["D"]],
    "ML_A": ml_probs[:, class_idx["A"]],
})

# ============================================================
# MERGE ALL THREE MODEL PREDICTIONS
# ============================================================

merged = elo.merge(poisson, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
merged = merged.merge(ml_df, on=["Date", "HomeTeam", "AwayTeam"], how="inner")

print("=" * 70)
print("ENSEMBLE EVALUATION (2025/26 final test)")
print("=" * 70)
print()
print(f"Merged rows: {len(merged)}")
print()

y = merged["FTR"].values

def score(probs_h, probs_d, probs_a, label):
    prob_matrix = np.column_stack([probs_a, probs_d, probs_h])
    ll = log_loss(y, prob_matrix, labels=["A", "D", "H"])
    preds = np.array(["A", "D", "H"])[np.argmax(prob_matrix, axis=1)]
    acc = accuracy_score(y, preds)
    print(f"  {label:26s} acc {acc:.2%}  logloss {ll:.4f}")
    return acc, ll

print("Individual models:")
elo_acc, elo_ll = score(merged["Elo_H"].values, merged["Elo_D"].values, merged["Elo_A"].values, "Elo 1X2")
poi_acc, poi_ll = score(merged["Poisson_H"].values, merged["Poisson_D"].values, merged["Poisson_A"].values, "Poisson + DC")
ml_acc, ml_ll = score(merged["ML_H"].values, merged["ML_D"].values, merged["ML_A"].values, "ML #4")
print()

# ============================================================
# EQUAL-WEIGHT BLEND
# ============================================================

blend_h = (merged["Elo_H"] + merged["Poisson_H"] + merged["ML_H"]) / 3
blend_d = (merged["Elo_D"] + merged["Poisson_D"] + merged["ML_D"]) / 3
blend_a = (merged["Elo_A"] + merged["Poisson_A"] + merged["ML_A"]) / 3

print("Equal-weight blend:")
blend_acc, blend_ll = score(blend_h.values, blend_d.values, blend_a.values, "Blend (1/3 each)")
print()

# ============================================================
# VERDICT
# ============================================================

best_individual_ll = min(elo_ll, poi_ll, ml_ll)
delta = blend_ll - best_individual_ll

print("=" * 70)
print("VERDICT")
print("=" * 70)
print()
print(f"Best individual log loss: {best_individual_ll:.4f}")
print(f"Blend log loss:           {blend_ll:.4f}")
print(f"Delta (blend - best):     {delta:+.4f}")
print()

if delta < -0.005:
    print(">> Ensemble beats best individual model.")
    print("   Worth pursuing weighted blends / stacking.")
elif delta < 0.005:
    print(">> Ensemble roughly ties best individual model.")
    print("   No clear gain from simple averaging.")
else:
    print(">> Ensemble is worse than best individual model.")
    print("   Models are not complementary enough to blend simply.")

print()
print("Market benchmark for reference: acc 49.47%  logloss 1.0153")