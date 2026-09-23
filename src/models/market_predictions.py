import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# ============================================================
# CONFIG
# ============================================================

MASTER_FILE = "data/premier_league_master.csv"

TRAIN_END = "2025-08-01"

DECAY = 0.003
REGULARIZATION = 0.01
MAX_ITER = 500
SCORE_MATRIX_SIZE = 10
RHO_BOUNDS = (-0.20, 0.20)

# ============================================================
# LOAD
# ============================================================

if not os.path.exists(MASTER_FILE):
    print(f"ERROR: missing {MASTER_FILE}")
    sys.exit(1)

df = pd.read_csv(MASTER_FILE)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

train = df[df["Date"] < TRAIN_END].copy()
test = df[df["Date"] >= TRAIN_END].copy()

print("=" * 70)
print("MARKET PREDICTIONS — EPL 2025/26")
print("=" * 70)
print(f"Training matches: {len(train)}")
print(f"Testing matches:  {len(test)}")
print()

# ============================================================
# FIT POISSON (time-decay + Dixon-Coles)
# ============================================================

def fit_poisson(training):
    teams = sorted(set(training["HomeTeam"]) | set(training["AwayTeam"]))
    t2i = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = training["HomeTeam"].map(t2i).to_numpy(int)
    ai = training["AwayTeam"].map(t2i).to_numpy(int)
    hg = training["FTHG"].to_numpy(float)
    ag = training["FTAG"].to_numpy(float)

    dates = training["Date"].to_numpy("datetime64[D]")
    latest = dates.max()
    age = (latest - dates).astype("timedelta64[D]").astype(float)

    cf = gammaln(hg + 1) + gammaln(ag + 1)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def unpack(p):
        return np.exp(p[:n]), np.exp(p[n:2*n]), np.exp(p[-2]), p[-1]

    def obj(p, w):
        atk, dfn, ha, rho = unpack(p)
        hl = np.clip(ha * atk[hi] * dfn[ai], 1e-8, None)
        al = np.clip(atk[ai] * dfn[hi], 1e-8, None)
        base = hg * np.log(hl) - hl + ag * np.log(al) - al - cf
        tau = np.ones_like(hg)
        tau[m00] = 1 - hl[m00] * al[m00] * rho
        tau[m01] = 1 + hl[m01] * rho
        tau[m10] = 1 + al[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)
        ll = base + np.log(tau)
        loss = -np.sum(w * ll)
        reg = REGULARIZATION * (
            np.sum(p[:n]**2) + np.sum(p[n:2*n]**2) + p[-2]**2 + p[-1]**2
        )
        return loss + reg

    w = np.exp(-DECAY * age)
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [np.log(1.10), 0.0]])
    bounds = [(None, None)] * (2*n) + [(None, None), RHO_BOUNDS]

    res = minimize(obj, x0, args=(w,), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": MAX_ITER, "ftol": 1e-10,
                            "gtol": 1e-6, "maxls": 40})
    if not res.success:
        raise RuntimeError(f"Poisson failed: {res.message}")

    atk, dfn, ha, rho = unpack(res.x)
    return {"atk": atk, "dfn": dfn, "ha": ha, "rho": rho, "t2i": t2i}

print("Fitting Poisson model...")
model = fit_poisson(train)
print(f"  Home advantage: {model['ha']:.3f}")
print(f"  Rho (Dixon-Coles): {model['rho']:+.4f}")
print()

# ============================================================
# SCORE MATRIX
# ============================================================

def score_matrix(hl, al, rho):
    g = np.arange(SCORE_MATRIX_SIZE)
    hp = np.exp(-hl + g * np.log(max(hl, 1e-8)) - gammaln(g + 1))
    ap = np.exp(-al + g * np.log(max(al, 1e-8)) - gammaln(g + 1))
    m = np.outer(hp, ap)
    m[0, 0] *= max(1 - hl * al * rho, 1e-10)
    m[0, 1] *= max(1 + hl * rho, 1e-10)
    m[1, 0] *= max(1 + al * rho, 1e-10)
    m[1, 1] *= max(1 - rho, 1e-10)
    return m / m.sum()

# ============================================================
# DERIVE ALL MARKETS FROM SCORE MATRIX
# ============================================================

def derive_markets(m):
    """Given a score matrix M[i,j], return dict of market probabilities."""
    goals = np.arange(SCORE_MATRIX_SIZE)
    total = goals[:, None] + goals[None, :]

    home_win = np.tril(m, -1).sum()
    draw = np.trace(m)
    away_win = np.triu(m, 1).sum()

    over_1_5 = m[total >= 2].sum()
    over_2_5 = m[total >= 3].sum()
    over_3_5 = m[total >= 4].sum()

    btts_yes = m[1:, 1:].sum()

    dc_1x = home_win + draw
    dc_x2 = draw + away_win
    dc_12 = home_win + away_win

    dnb_home = home_win / (home_win + away_win) if (home_win + away_win) > 0 else 0.5
    dnb_away = away_win / (home_win + away_win) if (home_win + away_win) > 0 else 0.5

    home_scores = 1 - m[0, :].sum()
    away_scores = 1 - m[:, 0].sum()

    return {
        "Over_1_5": over_1_5,
        "Over_2_5": over_2_5,
        "Over_3_5": over_3_5,
        "Under_1_5": 1 - over_1_5,
        "Under_2_5": 1 - over_2_5,
        "Under_3_5": 1 - over_3_5,
        "BTTS_Yes": btts_yes,
        "BTTS_No": 1 - btts_yes,
        "DC_1X": dc_1x,
        "DC_X2": dc_x2,
        "DC_12": dc_12,
        "DNB_Home": dnb_home,
        "DNB_Away": dnb_away,
        "Home_Team_To_Score": home_scores,
        "Away_Team_To_Score": away_scores,
    }

# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

atk, dfn, ha, rho = model["atk"], model["dfn"], model["ha"], model["rho"]
t2i = model["t2i"]

rows = []
for _, match in test.iterrows():
    h, a = match["HomeTeam"], match["AwayTeam"]
    h_atk = atk[t2i[h]] if h in t2i else 1.0
    h_dfn = dfn[t2i[h]] if h in t2i else 1.0
    a_atk = atk[t2i[a]] if a in t2i else 1.0
    a_dfn = dfn[t2i[a]] if a in t2i else 1.0

    hl = max(ha * h_atk * a_dfn, 1e-8)
    al = max(a_atk * h_dfn, 1e-8)

    m = score_matrix(hl, al, rho)
    markets = derive_markets(m)

    row = {
        "Date": match["Date"],
        "HomeTeam": h,
        "AwayTeam": a,
        "FTHG": int(match["FTHG"]),
        "FTAG": int(match["FTAG"]),
        "FTR": match["FTR"],
        "ExpectedHomeGoals": hl,
        "ExpectedAwayGoals": al,
    }
    row.update(markets)
    rows.append(row)

preds = pd.DataFrame(rows)

# Actual outcomes
preds["Actual_Over_1_5"] = ((preds["FTHG"] + preds["FTAG"]) >= 2).astype(int)
preds["Actual_Over_2_5"] = ((preds["FTHG"] + preds["FTAG"]) >= 3).astype(int)
preds["Actual_Over_3_5"] = ((preds["FTHG"] + preds["FTAG"]) >= 4).astype(int)
preds["Actual_BTTS_Yes"] = ((preds["FTHG"] >= 1) & (preds["FTAG"] >= 1)).astype(int)
preds["Actual_Home_Win"] = (preds["FTR"] == "H").astype(int)
preds["Actual_Draw"] = (preds["FTR"] == "D").astype(int)
preds["Actual_Away_Win"] = (preds["FTR"] == "A").astype(int)
preds["Actual_DC_1X"] = ((preds["FTR"] == "H") | (preds["FTR"] == "D")).astype(int)
preds["Actual_DC_X2"] = ((preds["FTR"] == "D") | (preds["FTR"] == "A")).astype(int)
preds["Actual_DC_12"] = ((preds["FTR"] == "H") | (preds["FTR"] == "A")).astype(int)
preds["Actual_Home_Team_To_Score"] = (preds["FTHG"] >= 1).astype(int)
preds["Actual_Away_Team_To_Score"] = (preds["FTAG"] >= 1).astype(int)

# ============================================================
# EVALUATE EACH MARKET
# ============================================================

print("=" * 70)
print("MARKET EVALUATION — 2025/26 TEST SEASON")
print("=" * 70)
print()

def evaluate_binary(probs, actuals, label):
    # Argmax prediction (threshold 0.5)
    preds_binary = (probs >= 0.5).astype(int)
    acc = accuracy_score(actuals, preds_binary)
    base_rate = actuals.mean()

    ll = log_loss(actuals, np.clip(probs, 1e-10, 1 - 1e-10), labels=[0, 1])
    brier = brier_score_loss(actuals, probs)

    print(f"  {label}")
    print(f"    Base rate (always majority): {max(base_rate, 1-base_rate):.1%}")
    print(f"    Model accuracy:              {acc:.1%}")
    print(f"    Log loss:                    {ll:.4f}")
    print(f"    Brier score:                 {brier:.4f}")

    # Confidence tiers
    confidence = np.maximum(probs, 1 - probs)
    n = len(confidence)
    for pct in [0.10, 0.25, 0.50]:
        k = int(n * pct)
        idx = np.argsort(confidence)[::-1][:k]
        tier_acc = accuracy_score(actuals.iloc[idx] if hasattr(actuals, 'iloc') else actuals[idx],
                                   preds_binary[idx])
        print(f"    Top {pct:.0%} confidence ({k} picks): {tier_acc:.1%}")

    print()

# 1X2 baseline
print("--- 1X2 (baseline) ---")
p_h = preds["DC_1X"].values - preds["Actual_Draw"].values * 0  # placeholder
# Actually recompute 1X2 properly
home_win_probs = preds["DC_1X"].values - preds["Actual_Draw"].values  # wrong
# Let me use a different approach: we didn't store 1X2 directly, so derive from DC
# DC_1X = home_win + draw; DC_X2 = draw + away_win; DC_12 = home_win + away_win
# We can solve: home_win = (DC_1X + DC_12 - DC_X2) / 2
hw = (preds["DC_1X"] + preds["DC_12"] - preds["DC_X2"]) / 2
dr = preds["DC_1X"] - hw
aw = preds["DC_12"] - hw
print(f"  1X2 accuracy: {accuracy_score(preds['FTR'], np.where(hw > dr, np.where(hw > aw, 'H', 'A'), np.where(dr > aw, 'D', 'A'))):.1%}")
print()

evaluate_binary(preds["Over_1_5"].values, preds["Actual_Over_1_5"], "Over 1.5 goals")
evaluate_binary(preds["Over_2_5"].values, preds["Actual_Over_2_5"], "Over 2.5 goals")
evaluate_binary(preds["Over_3_5"].values, preds["Actual_Over_3_5"], "Over 3.5 goals")
evaluate_binary(preds["BTTS_Yes"].values, preds["Actual_BTTS_Yes"], "BTTS (Yes)")
evaluate_binary(preds["DC_1X"].values, preds["Actual_DC_1X"], "Double Chance 1X (home or draw)")
evaluate_binary(preds["DC_X2"].values, preds["Actual_DC_X2"], "Double Chance X2 (draw or away)")
evaluate_binary(preds["DC_12"].values, preds["Actual_DC_12"], "Double Chance 12 (home or away)")
evaluate_binary(preds["DNB_Home"].values, preds["Actual_Home_Win"], "Draw No Bet (home)")
evaluate_binary(preds["Home_Team_To_Score"].values, preds["Actual_Home_Team_To_Score"], "Home Team to Score")
evaluate_binary(preds["Away_Team_To_Score"].values, preds["Actual_Away_Team_To_Score"], "Away Team to Score")

# ============================================================
# SAVE
# ============================================================

os.makedirs("data", exist_ok=True)
out_file = "data/market_predictions_2025_26.csv"
preds.to_csv(out_file, index=False)
print(f"Saved to: {out_file}")