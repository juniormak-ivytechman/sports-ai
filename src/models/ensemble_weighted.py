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
# GENERATE PREDICTIONS FOR A PERIOD
# ============================================================

def generate_predictions(period_start, period_end, train_end):
    period_df = df[(df["Date"] >= period_start) & (df["Date"] < period_end)].copy()

    # Elo
    elo_all = build_elo(df)
    elo_cal = elo_all[elo_all["Date"] < train_end].copy()
    elo_targets = elo_all[
        (elo_all["Date"] >= period_start) & (elo_all["Date"] < period_end)
    ].copy()
    elo_preds = elo_predict(elo_cal, elo_targets)

    # Poisson
    training = df[df["Date"] < train_end].copy()
    poi_model = fit_poisson(training)
    poi_preds = poisson_predict(poi_model, period_df)

    # ML
    features = prep_features(FEATURES_FILE)
    ml_train = features[features["Date"] < train_end].copy()
    ml_targets = features[
        (features["Date"] >= period_start) & (features["Date"] < period_end)
    ].copy()
    ml_model = fit_ml(ml_train)
    ml_preds = ml_predict(ml_model, ml_targets)

    merged = elo_preds.merge(poi_preds, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    merged = merged.merge(ml_preds, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    return merged


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("WEIGHTED ENSEMBLE — WEIGHTS TUNED ON VALIDATION")
print("=" * 70)
print()

val_start = pd.Timestamp(DEV_END)
val_end = pd.Timestamp(VAL_END)
test_start = pd.Timestamp(VAL_END)
test_end = pd.Timestamp("2030-01-01")

print("Generating validation predictions...")
val_preds = generate_predictions(val_start, val_end, DEV_END)
print(f"  Validation rows: {len(val_preds)}")

y_val = val_preds["FTR"].values


def get_ll(h, d, a, y):
    return log_loss(y, np.column_stack([a, d, h]), labels=["A", "D", "H"])


def get_acc(h, d, a, y):
    probs = np.column_stack([a, d, h])
    preds = np.array(["A", "D", "H"])[np.argmax(probs, axis=1)]
    return accuracy_score(y, preds)


print()
print("Validation — individual models:")
print(f"  Elo:      acc {get_acc(val_preds['Elo_H'], val_preds['Elo_D'], val_preds['Elo_A'], y_val):.2%}  "
      f"logloss {get_ll(val_preds['Elo_H'], val_preds['Elo_D'], val_preds['Elo_A'], y_val):.4f}")
print(f"  Poisson:  acc {get_acc(val_preds['Poisson_H'], val_preds['Poisson_D'], val_preds['Poisson_A'], y_val):.2%}  "
      f"logloss {get_ll(val_preds['Poisson_H'], val_preds['Poisson_D'], val_preds['Poisson_A'], y_val):.4f}")
print(f"  ML:       acc {get_acc(val_preds['ML_H'], val_preds['ML_D'], val_preds['ML_A'], y_val):.2%}  "
      f"logloss {get_ll(val_preds['ML_H'], val_preds['ML_D'], val_preds['ML_A'], y_val):.4f}")
print()

best_loss = 1e9
best_w = (1/3, 1/3, 1/3)
for we in np.arange(0, 1.01, 0.05):
    for wp in np.arange(0, 1.01 - we + 1e-9, 0.05):
        wm = 1.0 - we - wp
        h = we * val_preds["Elo_H"].values + wp * val_preds["Poisson_H"].values + wm * val_preds["ML_H"].values
        d = we * val_preds["Elo_D"].values + wp * val_preds["Poisson_D"].values + wm * val_preds["ML_D"].values
        a = we * val_preds["Elo_A"].values + wp * val_preds["Poisson_A"].values + wm * val_preds["ML_A"].values
        ll = get_ll(h, d, a, y_val)
        if ll < best_loss:
            best_loss = ll
            best_w = (we, wp, wm)

we, wp, wm = best_w

print("Optimal weights (grid search on validation):")
print(f"  Elo:     {we:.2f}")
print(f"  Poisson: {wp:.2f}")
print(f"  ML:      {wm:.2f}")
print(f"  Validation log loss: {best_loss:.4f}")
print()

print("Generating test predictions...")
test_preds = generate_predictions(test_start, test_end, VAL_END)
print(f"  Test rows: {len(test_preds)}")

y_test = test_preds["FTR"].values

h_t = we * test_preds["Elo_H"].values + wp * test_preds["Poisson_H"].values + wm * test_preds["ML_H"].values
d_t = we * test_preds["Elo_D"].values + wp * test_preds["Poisson_D"].values + wm * test_preds["ML_D"].values
a_t = we * test_preds["Elo_A"].values + wp * test_preds["Poisson_A"].values + wm * test_preds["ML_A"].values

print()
print("=" * 70)
print("FINAL TEST (2025/26)")
print("=" * 70)
print()
print(f"  Elo alone:      acc {get_acc(test_preds['Elo_H'], test_preds['Elo_D'], test_preds['Elo_A'], y_test):.2%}  "
      f"logloss {get_ll(test_preds['Elo_H'], test_preds['Elo_D'], test_preds['Elo_A'], y_test):.4f}")
print(f"  Poisson alone:  acc {get_acc(test_preds['Poisson_H'], test_preds['Poisson_D'], test_preds['Poisson_A'], y_test):.2%}  "
      f"logloss {get_ll(test_preds['Poisson_H'], test_preds['Poisson_D'], test_preds['Poisson_A'], y_test):.4f}")
print(f"  ML alone:       acc {get_acc(test_preds['ML_H'], test_preds['ML_D'], test_preds['ML_A'], y_test):.2%}  "
      f"logloss {get_ll(test_preds['ML_H'], test_preds['ML_D'], test_preds['ML_A'], y_test):.4f}")
print()
print(f"  Weighted blend (Elo {we:.2f}, Poisson {wp:.2f}, ML {wm:.2f}):")
print(f"    acc {get_acc(h_t, d_t, a_t, y_test):.2%}  logloss {get_ll(h_t, d_t, a_t, y_test):.4f}")
print()
print("  Reference: market acc 49.47%  logloss 1.0153")
print()

os.makedirs("data", exist_ok=True)
out = pd.DataFrame({
    "Date": test_preds["Date"],
    "HomeTeam": test_preds["HomeTeam"],
    "AwayTeam": test_preds["AwayTeam"],
    "FTR": test_preds["FTR"],
    "Elo_H": test_preds["Elo_H"], "Elo_D": test_preds["Elo_D"], "Elo_A": test_preds["Elo_A"],
    "Poisson_H": test_preds["Poisson_H"], "Poisson_D": test_preds["Poisson_D"], "Poisson_A": test_preds["Poisson_A"],
    "ML_H": test_preds["ML_H"], "ML_D": test_preds["ML_D"], "ML_A": test_preds["ML_A"],
    "Blend_H": h_t, "Blend_D": d_t, "Blend_A": a_t,
})
out.to_csv("data/ensemble_weighted_2025_26.csv", index=False)
print("Saved to: data/ensemble_weighted_2025_26.csv")