import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# ============================================================
# CONFIG
# ============================================================

MASTER_FILE = "data/premier_league_master.csv"

HOLDOUT_N = 100  # last N matches of the dataset are the "live" test

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

train = df.iloc[:-HOLDOUT_N].copy()
holdout = df.iloc[-HOLDOUT_N:].copy()

print("=" * 70)
print("LIVE SIMULATION")
print("=" * 70)
print(f"Fitted on: {len(train)} matches")
print(f"Last training match date: {train['Date'].max().date()}")
print(f"Held out: {len(holdout)} matches")
print(f"Holdout period: {holdout['Date'].min().date()} to {holdout['Date'].max().date()}")
print()

# ============================================================
# FIT POISSON
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

print("Fitting model on pre-holdout data...")
model = fit_poisson(train)
print(f"  Home advantage: {model['ha']:.3f}")
print(f"  Rho: {model['rho']:+.4f}")
print()

# ============================================================
# PREDICT HOLDOUT
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

def derive_markets(m):
    goals = np.arange(SCORE_MATRIX_SIZE)
    total = goals[:, None] + goals[None, :]

    home_win = np.tril(m, -1).sum()
    draw = np.trace(m)
    away_win = np.triu(m, 1).sum()

    return {
        "1X2_H": home_win,
        "1X2_D": draw,
        "1X2_A": away_win,
        "Over_1_5": m[total >= 2].sum(),
        "Over_2_5": m[total >= 3].sum(),
        "BTTS": m[1:, 1:].sum(),
        "DC_1X": home_win + draw,
        "DC_X2": draw + away_win,
        "DC_12": home_win + away_win,
        "Home_Score": 1 - m[0, :].sum(),
        "Away_Score": 1 - m[:, 0].sum(),
    }

atk, dfn, ha, rho = model["atk"], model["dfn"], model["ha"], model["rho"]
t2i = model["t2i"]

print("=" * 70)
print(f"PREDICTIONS FOR LAST {HOLDOUT_N} MATCHES (LIVE SIMULATION)")
print("=" * 70)
print()

correct = {
    "1X2": 0, "Over_1_5": 0, "Over_2_5": 0, "BTTS": 0,
    "DC_1X": 0, "DC_X2": 0, "DC_12": 0,
    "Home_Score": 0, "Away_Score": 0,
}

for i, (_, match) in enumerate(holdout.iterrows()):
    h, a = match["HomeTeam"], match["AwayTeam"]
    h_atk = atk[t2i[h]] if h in t2i else 1.0
    h_dfn = dfn[t2i[h]] if h in t2i else 1.0
    a_atk = atk[t2i[a]] if a in t2i else 1.0
    a_dfn = dfn[t2i[a]] if a in t2i else 1.0

    hl = max(ha * h_atk * a_dfn, 1e-8)
    al = max(a_atk * h_dfn, 1e-8)
    m = score_matrix(hl, al, rho)
    markets = derive_markets(m)

    fthg, ftag, ftr = int(match["FTHG"]), int(match["FTAG"]), match["FTR"]
    total = fthg + ftag

    # Actual outcomes
    actual = {
        "Over_1_5": total >= 2,
        "Over_2_5": total >= 3,
        "BTTS": fthg >= 1 and ftag >= 1,
        "DC_1X": ftr in ("H", "D"),
        "DC_X2": ftr in ("D", "A"),
        "DC_12": ftr in ("H", "A"),
        "Home_Score": fthg >= 1,
        "Away_Score": ftag >= 1,
    }

    # 1X2 prediction
    p_h, p_d, p_a = markets["1X2_H"], markets["1X2_D"], markets["1X2_A"]
    pred_1x2 = "H" if p_h >= max(p_d, p_a) else ("D" if p_d >= p_a else "A")

    # Which markets would we predict "yes"?
    def pick(prob, label):
        return "YES" if prob >= 0.5 else "NO"

    # Tally
    if pred_1x2 == ftr:
        correct["1X2"] += 1
    for mk, key in [("Over_1_5", "Over_1_5"), ("Over_2_5", "Over_2_5"),
                    ("BTTS", "BTTS"), ("DC_1X", "DC_1X"), ("DC_X2", "DC_X2"),
                    ("DC_12", "DC_12"), ("Home_Score", "Home_Score"),
                    ("Away_Score", "Away_Score")]:
        pred_yes = markets[mk] >= 0.5
        if pred_yes == actual[key]:
            correct[mk] += 1

    # Print match card
    date_str = match["Date"].date()
    print(f"--- Match {i+1}: {date_str} | {h} vs {a} ---")
    print(f"    Final score: {fthg}–{ftag}  ({ftr})")
    print(f"    Expected goals: {hl:.2f} – {al:.2f}")
    print()
    print(f"    1X2:    H {p_h:.1%}  D {p_d:.1%}  A {p_a:.1%}    → predicted {pred_1x2}, actual {ftr}  {'✓' if pred_1x2 == ftr else '✗'}")
    print()
    print(f"    Market predictions (P(yes) → prediction | actual):")

    market_display = [
        ("Over 1.5", "Over_1_5", actual["Over_1_5"]),
        ("Over 2.5", "Over_2_5", actual["Over_2_5"]),
        ("BTTS", "BTTS", actual["BTTS"]),
        ("DC 1X (H or D)", "DC_1X", actual["DC_1X"]),
        ("DC X2 (D or A)", "DC_X2", actual["DC_X2"]),
        ("DC 12 (H or A)", "DC_12", actual["DC_12"]),
        ("Home to Score", "Home_Score", actual["Home_Score"]),
        ("Away to Score", "Away_Score", actual["Away_Score"]),
    ]

    for label, mk, act in market_display:
        prob = markets[mk]
        pred_str = "YES" if prob >= 0.5 else "NO"
        act_str = "YES" if act else "NO"
        hit = "✓" if pred_str == act_str else "✗"
        print(f"      {label:20s} {prob:6.1%} → {pred_str:3s}  | actual {act_str:3s}  {hit}")

    print()

# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print(f"SUMMARY — {HOLDOUT_N} MATCHES")
print("=" * 70)
print()
for mk, count in correct.items():
    print(f"  {mk:14s}: {count}/{HOLDOUT_N} = {count/HOLDOUT_N:.0%}")
print()