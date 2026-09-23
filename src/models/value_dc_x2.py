import os
import sys
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_FILE = "data/market_predictions_2025_26.csv"
MASTER_FILE = "data/premier_league_master.csv"

# Assume bookmaker applies this margin on top of fair DC X2 odds.
# EPL DC markets typically run 4-8% margin. 5% is a conservative
# middle estimate.
BOOKMAKER_MARGIN = 0.05

# Thresholds to test (model edge over market)
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

# ============================================================
# LOAD
# ============================================================

for f in [PREDICTIONS_FILE, MASTER_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: missing {f}")
        sys.exit(1)

preds = pd.read_csv(PREDICTIONS_FILE)
master = pd.read_csv(MASTER_FILE)

preds["Date"] = pd.to_datetime(preds["Date"], format="%Y-%m-%d")
master["Date"] = pd.to_datetime(master["Date"], format="%Y-%m-%d")

# ============================================================
# MARKET PROBABILITIES (margin-normalized)
# ============================================================

master = master.dropna(subset=["AvgH", "AvgD", "AvgA"]).copy()

raw_h = 1.0 / master["AvgH"].astype(float)
raw_d = 1.0 / master["AvgD"].astype(float)
raw_a = 1.0 / master["AvgA"].astype(float)
total = raw_h + raw_d + raw_a

master["Mkt_H"] = raw_h / total
master["Mkt_D"] = raw_d / total
master["Mkt_A"] = raw_a / total
master["Mkt_X2"] = master["Mkt_D"] + master["Mkt_A"]

# ============================================================
# MERGE MODEL PREDICTIONS WITH MARKET
# ============================================================

merged = preds.merge(
    master[["Date", "HomeTeam", "AwayTeam", "Mkt_H", "Mkt_D", "Mkt_A", "Mkt_X2"]],
    on=["Date", "HomeTeam", "AwayTeam"],
    how="inner",
)

merged = merged.dropna(subset=["DC_X2", "Mkt_X2"]).copy()

# Actual outcome
merged["Actual_X2"] = merged["Actual_DC_X2"].astype(int)

# Edge = model minus market
merged["Edge"] = merged["DC_X2"] - merged["Mkt_X2"]

print("=" * 70)
print("DC X2 VALUE DETECTION — 2025/26")
print("=" * 70)
print(f"Matches with both model and market data: {len(merged)}")
print(f"Model avg P(X2):   {merged['DC_X2'].mean():.3f}")
print(f"Market avg P(X2):  {merged['Mkt_X2'].mean():.3f}")
print(f"Actual X2 rate:    {merged['Actual_X2'].mean():.3f}")
print()

# ============================================================
# SIMULATE
# ============================================================

def simulate(df):
    """Bet $1 on DC X2 for every row in df. Return (profit, stake, roi)."""
    profits = []
    for _, row in df.iterrows():
        # Offered odds = fair * (1 + margin) inversed
        offered_odds = 1.0 / (row["Mkt_X2"] * (1.0 + BOOKMAKER_MARGIN))
        if row["Actual_X2"] == 1:
            profits.append(offered_odds - 1.0)  # win
        else:
            profits.append(-1.0)  # lose
    profits = np.array(profits)
    stake = len(df)
    return profits.sum(), stake, profits.sum() / stake if stake > 0 else 0.0

# Baseline: bet everything
profit_all, stake_all, roi_all = simulate(merged)
print(f"--- Baseline: bet ALL {len(merged)} matches ---")
print(f"    Total profit: {profit_all:+.2f}")
print(f"    Total staked: {stake_all:.2f}")
print(f"    ROI:          {roi_all:+.2%}")
print()

# Threshold search
print("=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)
print()
print(f"{'Thresh':>8s} | {'N':>4s} | {'Model':>7s} | {'Market':>7s} | {'Actual':>7s} | {'ROI':>8s} | {'Profit':>8s}")
print("-" * 74)

for t in THRESHOLDS:
    flagged = merged[merged["Edge"] >= t]
    if len(flagged) == 0:
        print(f"{t:>8.2%} |    0 |       - |       - |       - |        - |        -")
        continue
    profit, stake, roi = simulate(flagged)
    print(f"{t:>8.2%} | {len(flagged):>4d} | "
          f"{flagged['DC_X2'].mean():>7.3f} | "
          f"{flagged['Mkt_X2'].mean():>7.3f} | "
          f"{flagged['Actual_X2'].mean():>7.3f} | "
          f"{roi:>+8.2%} | "
          f"{profit:>+8.2f}")

print()

# ============================================================
# HONESTY CHECK: what does the model predict vs what happened
# ============================================================

print("=" * 70)
print("CALIBRATION CHECK ON FLAGGED SUBSETS")
print("=" * 70)
print()
print("If the model is right, 'Actual' should be close to 'Model'.")
print("If the market is right, 'Actual' should be close to 'Market'.")
print()

for t in THRESHOLDS:
    flagged = merged[merged["Edge"] >= t]
    if len(flagged) < 10:
        continue
    model_avg = flagged["DC_X2"].mean()
    market_avg = flagged["Mkt_X2"].mean()
    actual = flagged["Actual_X2"].mean()

    model_err = abs(model_avg - actual)
    market_err = abs(market_avg - actual)

    winner = "MODEL" if model_err < market_err else "MARKET"
    print(f"  Edge ≥ {t:.0%}: N={len(flagged):>3d}  "
          f"model_err={model_err:.3f}  market_err={market_err:.3f}  → {winner} closer")

print()

# ============================================================
# BEST-OF-CHECK
# ============================================================

best_roi = -999
best_t = None
for t in THRESHOLDS:
    flagged = merged[merged["Edge"] >= t]
    if len(flagged) < 20:
        continue
    _, _, roi = simulate(flagged)
    if roi > best_roi:
        best_roi = roi
        best_t = t

print("=" * 70)
print("VERDICT")
print("=" * 70)
print()

if best_t is None:
    print("Not enough flagged matches to evaluate.")
elif best_roi > 0.02:
    print(f">> POSITIVE EDGE at threshold {best_t:.0%}")
    print(f"   ROI: {best_roi:+.2%}")
    print(f"   This is a genuine signal. Worth further investigation.")
elif best_roi > -0.02:
    print(f">> BREAK-EVEN at threshold {best_t:.0%}")
    print(f"   ROI: {best_roi:+.2%}")
    print(f"   Model is close to the market. No clear betting edge.")
else:
    print(f">> NEGATIVE at best threshold {best_t:.0%}")
    print(f"   ROI: {best_roi:+.2%}")
    print(f"   Market is too efficient on this signal. Not profitable.")

print()
print("Note: this assumes a fixed 5% bookmaker margin on DC X2.")
print("Real bookmaker margins vary. Treat these ROIs as estimates.")