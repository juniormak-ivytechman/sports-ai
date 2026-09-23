import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

MASTER_FILE = "data/premier_league_master.csv"
FEATURES_FILE = "data/premier_league_features.csv"

DEV_END = "2023-08-01"
VAL_END = "2025-08-01"

STARTING_RATING = 1500
K_FACTOR = 40
HOME_ADVANTAGE = 40
BAND_WIDTH = 75
MIN_MATCHES = 50

DECAY = 0.003
REGULARIZATION = 0.01
MAX_ITER = 500
SCORE_MATRIX_SIZE = 10
RHO_BOUNDS = (-0.20, 0.20)

GRID_STEP = 0.05

# ============================================================
# LOAD
# ============================================================

for f in [MASTER_FILE, FEATURES_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: missing {f}")
        sys.exit(1)

df = pd.read_csv(MASTER_FILE)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

# ============================================================
# MARKET PROBABILITIES
# ============================================================

def add_market_probs(frame):
    """Return a copy with margin-normalized market H/D/A probabilities.
    Rows missing any of AvgH/AvgD/AvgA are dropped."""
    frame = frame.dropna(subset=["AvgH", "AvgD", "AvgA"]).copy()
    raw_h = 1.0 / frame["AvgH"].astype(float)
    raw_d = 1.0 / frame["AvgD"].astype(float)
    raw_a = 1.0 / frame["AvgA"].astype(float)
    total = raw_h + raw_d + raw_a
    frame["Market_H"] = raw_h / total
    frame["Market_D"] = raw_d / total
    frame["Market_A"] = raw_a / total
    return frame

# ============================================================
# ELO
# ============================================================

def build_elo(matches):
    ratings = {}
    rows = []
    for _, m in matches.iterrows():
        h, a = m["HomeTeam"], m["AwayTeam"]
        hr = ratings.get(h, STARTING_RATING)
        ar = ratings.get(a, STARTING_RATING)
        diff = hr + HOME_ADVANTAGE - ar
        hp = 1.0 / (1.0 + 10 ** ((ar - hr - HOME_ADVANTAGE) / 400))
        rows.append({
            "Date": m["Date"], "HomeTeam": h, "AwayTeam": a,
            "FTR": m["FTR"], "EloDifference": diff,
        })
        if m["FTR"] == "H":
            ah, aa = 1.0, 0.0
        elif m["FTR"] == "D":
            ah, aa = 0.5, 0.5
        else:
            ah, aa = 0.0, 1.0
        ratings[h] = hr + K_FACTOR * (ah - hp)
        ratings[a] = ar + K_FACTOR * (aa - (1 - hp))
    return pd.DataFrame(rows)


def elo_predict(calibration, targets):
    def estimate(diff):
        band = BAND_WIDTH
        local = calibration[
            (calibration["EloDifference"] >= diff - band) &
            (calibration["EloDifference"] <= diff + band)
        ]
        while len(local) < MIN_MATCHES and band < 400:
            band += 25
            local = calibration[
                (calibration["EloDifference"] >= diff - band) &
                (calibration["EloDifference"] <= diff + band)
            ]
        h = (local["FTR"] == "H").sum()
        d = (local["FTR"] == "D").sum()
        a = (local["FTR"] == "A").sum()
        s = 1
        t = h + d + a + 3 * s
        return (h + s) / t, (d + s) / t, (a + s) / t

    rows = []
    for _, m in targets.iterrows():
        ph, pd_, pa = estimate(m["EloDifference"])
        rows.append({
            "Date": m["Date"], "HomeTeam": m["HomeTeam"], "AwayTeam": m["AwayTeam"],
            "FTR": m["FTR"], "Elo_H": ph, "Elo_D": pd_, "Elo_A": pa,
        })
    return pd.DataFrame(rows)

# ============================================================
# POISSON + DC
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


def score_mtx(hl, al, rho):
    g = np.arange(SCORE_MATRIX_SIZE)
    hp = np.exp(-hl + g * np.log(max(hl, 1e-8)) - gammaln(g + 1))
    ap = np.exp(-al + g * np.log(max(al, 1e-8)) - gammaln(g + 1))
    m = np.outer(hp, ap)
    m[0, 0] *= max(1 - hl * al * rho, 1e-10)
    m[0, 1] *= max(1 + hl * rho, 1e-10)
    m[1, 0] *= max(1 + al * rho, 1e-10)
    m[1, 1] *= max(1 - rho, 1e-10)
    return m / m.sum()


def poisson_predict(model, targets):
    atk, dfn, ha, rho = model["atk"], model["dfn"], model["ha"], model["rho"]
    t2i = model["t2i"]
    rows = []
    for _, m in targets.iterrows():
        h, a = m["HomeTeam"], m["AwayTeam"]
        h_atk = atk[t2i[h]] if h in t2i else 1.0
        h_dfn = dfn[t2i[h]] if h in t2i else 1.0
        a_atk = atk[t2i[a]] if a in t2i else 1.0
        a_dfn = dfn[t2i[a]] if a in t2i else 1.0
        hl = max(ha * h_atk * a_dfn, 1e-8)
        al = max(a_atk * h_dfn, 1e-8)
        mtx = score_mtx(hl, al, rho)
        ph = np.tril(mtx, -1).sum()
        pd_ = np.trace(mtx)
        pa = np.triu(mtx, 1).sum()
        rows.append({
            "Date": m["Date"], "HomeTeam": h, "AwayTeam": a, "FTR": m["FTR"],
            "Poisson_H": ph, "Poisson_D": pd_, "Poisson_A": pa,
        })
    return pd.DataFrame(rows)

# ============================================================
# ML #4
# ============================================================

FEATURE_COLS = [
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


def prep_features(path):
    fd = pd.read_csv(path)
    fd["Date"] = pd.to_datetime(fd["Date"], format="%Y-%m-%d")
    fd["FormPointsDifference"] = fd["HomeFormPoints"] - fd["AwayFormPoints"]
    fd["HomeAwayFormDifference"] = fd["HomeTeamHomeFormPoints"] - fd["AwayTeamAwayFormPoints"]
    fd["GoalsScoredDifference"] = fd["HomeAvgGoalsScored"] - fd["AwayAvgGoalsScored"]
    fd["GoalsConcededDifference"] = fd["HomeAvgGoalsConceded"] - fd["AwayAvgGoalsConceded"]
    fd["ShotsDifference"] = fd["HomeAvgShots"] - fd["AwayAvgShots"]
    fd["ShotsOnTargetDifference"] = fd["HomeAvgShotsOnTarget"] - fd["AwayAvgShotsOnTarget"]
    fd["CornersDifference"] = fd["HomeAvgCorners"] - fd["AwayAvgCorners"]
    return fd[(fd["HomeFormPoints"] > 0) | (fd["AwayFormPoints"] > 0)].copy()


def fit_ml(training):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(training[FEATURE_COLS], training["FTR"])
    return pipe


def ml_predict(model, targets):
    probs = model.predict_proba(targets[FEATURE_COLS])
    classes = model.named_steps["classifier"].classes_
    ci = {c: i for i, c in enumerate(classes)}
    return pd.DataFrame({
        "Date": targets["Date"].values,
        "HomeTeam": targets["HomeTeam"].values,
        "AwayTeam": targets["AwayTeam"].values,
        "FTR": targets["FTR"].values,
        "ML_H": probs[:, ci["H"]],
        "ML_D": probs[:, ci["D"]],
        "ML_A": probs[:, ci["A"]],
    })

# ============================================================
# GENERATE PREDICTIONS FOR A PERIOD (INCLUDING MARKET)
# ============================================================

def generate_predictions(period_start, period_end, train_end):
    period_df = df[(df["Date"] >= period_start) & (df["Date"] < period_end)].copy()

    elo_all = build_elo(df)
    elo_cal = elo_all[elo_all["Date"] < train_end].copy()
    elo_targets = elo_all[
        (elo_all["Date"] >= period_start) & (elo_all["Date"] < period_end)
    ].copy()
    elo_preds = elo_predict(elo_cal, elo_targets)

    training = df[df["Date"] < train_end].copy()
    poi_model = fit_poisson(training)
    poi_preds = poisson_predict(poi_model, period_df)

    features = prep_features(FEATURES_FILE)
    ml_train = features[features["Date"] < train_end].copy()
    ml_targets = features[
        (features["Date"] >= period_start) & (features["Date"] < period_end)
    ].copy()
    ml_model = fit_ml(ml_train)
    ml_preds = ml_predict(ml_model, ml_targets)

    # Market probabilities come from the master file
    market_df = add_market_probs(period_df)
    market_df = market_df[["Date", "HomeTeam", "AwayTeam",
                           "Market_H", "Market_D", "Market_A"]]

    merged = elo_preds.merge(poi_preds, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    merged = merged.merge(ml_preds, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    merged = merged.merge(market_df, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    return merged

# ============================================================
# HELPERS
# ============================================================

def acc_and_ll(probs_h, probs_d, probs_a, y):
    probs = np.column_stack([probs_a, probs_d, probs_h])  # [A, D, H] order
    preds = np.array(["A", "D", "H"])[np.argmax(probs, axis=1)]
    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs, labels=["A", "D", "H"])
    return acc, ll

# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FOUR-WAY ENSEMBLE (Elo + Poisson + ML + Market)")
print("=" * 70)
print()

val_start = pd.Timestamp(DEV_END)
val_end = pd.Timestamp(VAL_END)
test_start = pd.Timestamp(VAL_END)
test_end = pd.Timestamp("2030-01-01")

print("Generating validation predictions...")
val_preds = generate_predictions(val_start, val_end, DEV_END)
print(f"  Validation rows (all four models + market): {len(val_preds)}")
y_val = val_preds["FTR"].values

# Individual models on validation
elo_acc_v, elo_ll_v = acc_and_ll(val_preds["Elo_H"], val_preds["Elo_D"], val_preds["Elo_A"], y_val)
poi_acc_v, poi_ll_v = acc_and_ll(val_preds["Poisson_H"], val_preds["Poisson_D"], val_preds["Poisson_A"], y_val)
ml_acc_v,  ml_ll_v  = acc_and_ll(val_preds["ML_H"], val_preds["ML_D"], val_preds["ML_A"], y_val)
mkt_acc_v, mkt_ll_v = acc_and_ll(val_preds["Market_H"], val_preds["Market_D"], val_preds["Market_A"], y_val)

print()
print("Validation — individual models:")
print(f"  Elo:      acc {elo_acc_v:.2%}  logloss {elo_ll_v:.4f}")
print(f"  Poisson:  acc {poi_acc_v:.2%}  logloss {poi_ll_v:.4f}")
print(f"  ML:       acc {ml_acc_v:.2%}  logloss {ml_ll_v:.4f}")
print(f"  Market:   acc {mkt_acc_v:.2%}  logloss {mkt_ll_v:.4f}")
print()

# Pre-stack: (N, 4, 3) with order [Elo, Poisson, ML, Market], outcome axis [H, D, A]
stack_val = np.stack([
    val_preds[["Elo_H", "Elo_D", "Elo_A"]].to_numpy(),
    val_preds[["Poisson_H", "Poisson_D", "Poisson_A"]].to_numpy(),
    val_preds[["ML_H", "ML_D", "ML_A"]].to_numpy(),
    val_preds[["Market_H", "Market_D", "Market_A"]].to_numpy(),
], axis=1)

# Label index for log loss: H=0, D=1, A=2
label_to_idx = {"H": 0, "D": 1, "A": 2}
y_val_idx = np.array([label_to_idx[x] for x in y_val])

# Grid search 4 weights, step GRID_STEP, summing to 1
steps = int(round(1.0 / GRID_STEP))
best_ll = 1e9
best_w = (0.25, 0.25, 0.25, 0.25)

for i in range(steps + 1):
    we = i * GRID_STEP
    for j in range(steps + 1 - i):
        wp = j * GRID_STEP
        for k in range(steps + 1 - i - j):
            wm = k * GRID_STEP
            wk = 1.0 - we - wp - wm
            if wk < -1e-9:
                continue
            w = np.array([we, wp, wm, wk])
            blended = np.tensordot(w, stack_val, axes=([0], [1]))  # (N, 3)
            # Normalize in case of floating point drift
            blended = blended / blended.sum(axis=1, keepdims=True)
            # log loss for the outcome that actually happened
            correct_probs = blended[np.arange(len(y_val_idx)), y_val_idx]
            correct_probs = np.clip(correct_probs, 1e-15, 1.0)
            ll = -np.mean(np.log(correct_probs))
            if ll < best_ll:
                best_ll = ll
                best_w = (we, wp, wm, wk)

we, wp, wm, wk = best_w

print("Optimal four-way weights (grid search on validation):")
print(f"  Elo:     {we:.2f}")
print(f"  Poisson: {wp:.2f}")
print(f"  ML:      {wm:.2f}")
print(f"  Market:  {wk:.2f}")
print(f"  Validation log loss (4-way): {best_ll:.4f}")
print()

# Also report the best 3-way (no market) blend on the same rows for comparison
best_ll_3 = 1e9
best_w_3 = (1/3, 1/3, 1/3)
for i in range(steps + 1):
    we3 = i * GRID_STEP
    for j in range(steps + 1 - i):
        wp3 = j * GRID_STEP
        wm3 = 1.0 - we3 - wp3
        if wm3 < -1e-9:
            continue
        w3 = np.array([we3, wp3, wm3, 0.0])
        blended = np.tensordot(w3, stack_val, axes=([0], [1]))
        blended = blended / blended.sum(axis=1, keepdims=True)
        correct_probs = blended[np.arange(len(y_val_idx)), y_val_idx]
        correct_probs = np.clip(correct_probs, 1e-15, 1.0)
        ll = -np.mean(np.log(correct_probs))
        if ll < best_ll_3:
            best_ll_3 = ll
            best_w_3 = (we3, wp3, wm3)

print(f"Best three-way blend (no market), same rows: log loss {best_ll_3:.4f} "
      f"(Elo {best_w_3[0]:.2f}, Poisson {best_w_3[1]:.2f}, ML {best_w_3[2]:.2f})")
print()

print("Generating test predictions...")
test_preds = generate_predictions(test_start, test_end, VAL_END)
print(f"  Test rows (all four models + market): {len(test_preds)}")
y_test = test_preds["FTR"].values

elo_acc_t, elo_ll_t = acc_and_ll(test_preds["Elo_H"], test_preds["Elo_D"], test_preds["Elo_A"], y_test)
poi_acc_t, poi_ll_t = acc_and_ll(test_preds["Poisson_H"], test_preds["Poisson_D"], test_preds["Poisson_A"], y_test)
ml_acc_t,  ml_ll_t  = acc_and_ll(test_preds["ML_H"], test_preds["ML_D"], test_preds["ML_A"], y_test)
mkt_acc_t, mkt_ll_t = acc_and_ll(test_preds["Market_H"], test_preds["Market_D"], test_preds["Market_A"], y_test)

# Apply chosen 4-way weights to test
stack_test = np.stack([
    test_preds[["Elo_H", "Elo_D", "Elo_A"]].to_numpy(),
    test_preds[["Poisson_H", "Poisson_D", "Poisson_A"]].to_numpy(),
    test_preds[["ML_H", "ML_D", "ML_A"]].to_numpy(),
    test_preds[["Market_H", "Market_D", "Market_A"]].to_numpy(),
], axis=1)

w = np.array(best_w)
blend_test = np.tensordot(w, stack_test, axes=([0], [1]))
blend_test = blend_test / blend_test.sum(axis=1, keepdims=True)
blend_acc_t, blend_ll_t = acc_and_ll(blend_test[:, 0], blend_test[:, 1], blend_test[:, 2], y_test)

# Also test the 3-way weights (from validation) on the same rows
w3 = np.array([best_w_3[0], best_w_3[1], best_w_3[2], 0.0])
blend3_test = np.tensordot(w3, stack_test, axes=([0], [1]))
blend3_test = blend3_test / blend3_test.sum(axis=1, keepdims=True)
blend3_acc_t, blend3_ll_t = acc_and_ll(blend3_test[:, 0], blend3_test[:, 1], blend3_test[:, 2], y_test)

print()
print("=" * 70)
print("FINAL TEST (2025/26, subset with market odds)")
print("=" * 70)
print()
print(f"  Elo alone:               acc {elo_acc_t:.2%}  logloss {elo_ll_t:.4f}")
print(f"  Poisson alone:           acc {poi_acc_t:.2%}  logloss {poi_ll_t:.4f}")
print(f"  ML alone:                acc {ml_acc_t:.2%}  logloss {ml_ll_t:.4f}")
print(f"  Market alone:            acc {mkt_acc_t:.2%}  logloss {mkt_ll_t:.4f}")
print()
print(f"  Three-way blend (no market) [Elo {best_w_3[0]:.2f}, Poisson {best_w_3[1]:.2f}, ML {best_w_3[2]:.2f}]:")
print(f"    acc {blend3_acc_t:.2%}  logloss {blend3_ll_t:.4f}")
print()
print(f"  Four-way blend [Elo {we:.2f}, Poisson {wp:.2f}, ML {wm:.2f}, Market {wk:.2f}]:")
print(f"    acc {blend_acc_t:.2%}  logloss {blend_ll_t:.4f}")
print()

os.makedirs("data", exist_ok=True)
out = test_preds.copy()
out["Blend3_H"] = blend3_test[:, 0]
out["Blend3_D"] = blend3_test[:, 1]
out["Blend3_A"] = blend3_test[:, 2]
out["Blend4_H"] = blend_test[:, 0]
out["Blend4_D"] = blend_test[:, 1]
out["Blend4_A"] = blend_test[:, 2]
out.to_csv("data/ensemble_with_market_2025_26.csv", index=False)
print("Saved to: data/ensemble_with_market_2025_26.csv")