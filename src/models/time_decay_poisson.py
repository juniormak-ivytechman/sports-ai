import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, log_loss

# ============================================================
# CONFIG
# ============================================================

MASTER_FILE = "data/premier_league_master.csv"

DECAY_VALUES = [0.0030]

REGULARIZATION = 0.01
MAX_ITER = 500

UNKNOWN_ATTACK = 1.0
UNKNOWN_DEFENCE = 1.0

SCORE_MATRIX_SIZE = 10

RHO_BOUNDS = (-0.20, 0.20)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(MASTER_FILE):
    print(f"ERROR: Could not find {MASTER_FILE}")
    sys.exit(1)

df = pd.read_csv(MASTER_FILE)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
df = df.sort_values("Date").reset_index(drop=True)

required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"]
missing = [c for c in required if c not in df.columns]
if missing:
    print("ERROR: Missing columns:", missing)
    sys.exit(1)

development_df = df[df["Date"] < "2023-08-01"].copy()
validation_df = df[(df["Date"] >= "2023-08-01") & (df["Date"] < "2025-08-01")].copy()
final_test_df = df[df["Date"] >= "2025-08-01"].copy()

print("=" * 70)
print("TIME-DECAY POISSON + HOME/AWAY ATTACK/DEFENCE + DIXON-COLES")
print("=" * 70)
print(f"Development matches : {len(development_df)}")
print(f"Validation matches  : {len(validation_df)}")
print(f"Final test matches  : {len(final_test_df)}")
print(f"Decay values        : {DECAY_VALUES}")
print()

# ============================================================
# FIT ONE MODEL
# ============================================================

def fit_model(training_df, decay):
    teams = sorted(set(training_df["HomeTeam"]) | set(training_df["AwayTeam"]))
    team_to_index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = training_df["HomeTeam"].map(team_to_index).to_numpy(dtype=int)
    away_idx = training_df["AwayTeam"].map(team_to_index).to_numpy(dtype=int)
    home_goals = training_df["FTHG"].to_numpy(dtype=float)
    away_goals = training_df["FTAG"].to_numpy(dtype=float)

    train_dates = training_df["Date"].to_numpy(dtype="datetime64[D]")
    latest_date = train_dates.max()
    age_days = (latest_date - train_dates).astype("timedelta64[D]").astype(float)

    home_goal_log_factorial = gammaln(home_goals + 1)
    away_goal_log_factorial = gammaln(away_goals + 1)
    constant_log_factorial = home_goal_log_factorial + away_goal_log_factorial

    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    def unpack_params(params):
        home_attack_log = params[0:n_teams]
        home_defence_log = params[n_teams:2 * n_teams]
        away_attack_log = params[2 * n_teams:3 * n_teams]
        away_defence_log = params[3 * n_teams:4 * n_teams]
        rho = params[-1]
        return (
            np.exp(home_attack_log),
            np.exp(home_defence_log),
            np.exp(away_attack_log),
            np.exp(away_defence_log),
            rho,
        )

    def objective(params, weights):
        home_attack, home_defence, away_attack, away_defence, rho = unpack_params(params)

        home_lambda = home_attack[home_idx] * away_defence[away_idx]
        away_lambda = away_attack[away_idx] * home_defence[home_idx]

        home_lambda = np.clip(home_lambda, 1e-8, None)
        away_lambda = np.clip(away_lambda, 1e-8, None)

        base_log_likelihood = (
            home_goals * np.log(home_lambda) - home_lambda
            + away_goals * np.log(away_lambda) - away_lambda
            - constant_log_factorial
        )

        tau = np.ones_like(home_goals)
        tau[mask_00] = 1.0 - home_lambda[mask_00] * away_lambda[mask_00] * rho
        tau[mask_01] = 1.0 + home_lambda[mask_01] * rho
        tau[mask_10] = 1.0 + away_lambda[mask_10] * rho
        tau[mask_11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)

        log_likelihood = base_log_likelihood + np.log(tau)
        weighted_loss = -np.sum(weights * log_likelihood)

        regularization = REGULARIZATION * (
            np.sum(params[0:n_teams] ** 2)
            + np.sum(params[n_teams:2 * n_teams] ** 2)
            + np.sum(params[2 * n_teams:3 * n_teams] ** 2)
            + np.sum(params[3 * n_teams:4 * n_teams] ** 2)
            + params[-1] ** 2
        )

        return weighted_loss + regularization

    weights = np.exp(-decay * age_days)

    initial_params = np.concatenate([
        np.zeros(n_teams),   # home attack
        np.zeros(n_teams),   # home defence
        np.zeros(n_teams),   # away attack
        np.zeros(n_teams),   # away defence
        [0.0],               # rho
    ])

    bounds = [(None, None)] * (4 * n_teams) + [RHO_BOUNDS]

    result = minimize(
        objective,
        initial_params,
        args=(weights,),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": MAX_ITER,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 40,
        },
    )

    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")

    home_attack, home_defence, away_attack, away_defence, rho = unpack_params(result.x)

    return {
        "result": result,
        "home_attack": home_attack,
        "home_defence": home_defence,
        "away_attack": away_attack,
        "away_defence": away_defence,
        "rho": rho,
        "teams": teams,
        "team_to_index": team_to_index,
    }

# ============================================================
# SCORE PROBABILITIES
# ============================================================

def poisson_score_matrix(home_lambda, away_lambda, rho):
    goals = np.arange(SCORE_MATRIX_SIZE)
    home_pmf = np.exp(
        -home_lambda
        + goals * np.log(max(home_lambda, 1e-8))
        - gammaln(goals + 1)
    )
    away_pmf = np.exp(
        -away_lambda
        + goals * np.log(max(away_lambda, 1e-8))
        - gammaln(goals + 1)
    )
    matrix = np.outer(home_pmf, away_pmf)

    tau_00 = max(1.0 - home_lambda * away_lambda * rho, 1e-10)
    tau_01 = max(1.0 + home_lambda * rho, 1e-10)
    tau_10 = max(1.0 + away_lambda * rho, 1e-10)
    tau_11 = max(1.0 - rho, 1e-10)

    matrix[0, 0] *= tau_00
    matrix[0, 1] *= tau_01
    matrix[1, 0] *= tau_10
    matrix[1, 1] *= tau_11

    return matrix / matrix.sum()

# ============================================================
# PREDICT
# ============================================================

def predict(model, matches):
    home_attack = model["home_attack"]
    home_defence = model["home_defence"]
    away_attack = model["away_attack"]
    away_defence = model["away_defence"]
    rho = model["rho"]
    team_to_index = model["team_to_index"]

    rows = []

    for _, match in matches.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        if home_team in team_to_index:
            h = team_to_index[home_team]
            ht_home_attack = home_attack[h]
            ht_home_defence = home_defence[h]
            unknown_home = False
        else:
            ht_home_attack = UNKNOWN_ATTACK
            ht_home_defence = UNKNOWN_DEFENCE
            unknown_home = True

        if away_team in team_to_index:
            a = team_to_index[away_team]
            at_away_attack = away_attack[a]
            at_away_defence = away_defence[a]
            unknown_away = False
        else:
            at_away_attack = UNKNOWN_ATTACK
            at_away_defence = UNKNOWN_DEFENCE
            unknown_away = True

        home_lambda = max(ht_home_attack * at_away_defence, 1e-8)
        away_lambda = max(at_away_attack * ht_home_defence, 1e-8)

        matrix = poisson_score_matrix(home_lambda, away_lambda, rho)

        home_probability = np.tril(matrix, -1).sum()
        draw_probability = np.trace(matrix)
        away_probability = np.triu(matrix, 1).sum()

        probabilities = {"H": home_probability, "D": draw_probability, "A": away_probability}
        prediction = max(probabilities, key=probabilities.get)

        actual_home_goals = int(match["FTHG"])
        actual_away_goals = int(match["FTAG"])
        actual_score_probability = (
            matrix[actual_home_goals, actual_away_goals]
            if actual_home_goals < SCORE_MATRIX_SIZE and actual_away_goals < SCORE_MATRIX_SIZE
            else 0.0
        )

        rows.append({
            "Date": match["Date"],
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTR": match["FTR"],
            "HomeGoals": actual_home_goals,
            "AwayGoals": actual_away_goals,
            "ExpectedHomeGoals": home_lambda,
            "ExpectedAwayGoals": away_lambda,
            "HomeProbability": home_probability,
            "DrawProbability": draw_probability,
            "AwayProbability": away_probability,
            "Prediction": prediction,
            "ActualScoreProbability": actual_score_probability,
            "UnknownHomeTeam": unknown_home,
            "UnknownAwayTeam": unknown_away,
        })

    return pd.DataFrame(rows)

# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, matches):
    results = predict(model, matches)

    accuracy = accuracy_score(results["FTR"], results["Prediction"])

    probability_matrix = results[["AwayProbability", "DrawProbability", "HomeProbability"]].values
    model_log_loss = log_loss(results["FTR"], probability_matrix, labels=["A", "D", "H"])

    home_mae = (results["ExpectedHomeGoals"] - results["HomeGoals"]).abs().mean()
    away_mae = (results["ExpectedAwayGoals"] - results["AwayGoals"]).abs().mean()
    total_mae = (
        (results["ExpectedHomeGoals"] + results["ExpectedAwayGoals"])
        - (results["HomeGoals"] + results["AwayGoals"])
    ).abs().mean()

    mean_actual_score_probability = results["ActualScoreProbability"].mean()

    return {
        "results": results,
        "accuracy": accuracy,
        "log_loss": model_log_loss,
        "home_mae": home_mae,
        "away_mae": away_mae,
        "total_mae": total_mae,
        "mean_actual_score_probability": mean_actual_score_probability,
    }

# ============================================================
# VALIDATION + FINAL TEST
# ============================================================

print("=" * 70)
print("VALIDATION (home/away Poisson + DC)")
print("=" * 70)
print()

for decay in DECAY_VALUES:
    val_model = fit_model(development_df, decay)
    val_metrics = evaluate(val_model, validation_df)
    print(f"Decay = {decay:.4f} | rho = {val_model['rho']:+.4f} | "
          f"Validation accuracy = {val_metrics['accuracy']:.2%} | "
          f"Validation log loss = {val_metrics['log_loss']:.4f}")

print()
print("=" * 70)
print("FINAL 2025/26 TEST (home/away Poisson + DC)")
print("=" * 70)
print()

pre_test_df = df[df["Date"] < "2025-08-01"].copy()
best_decay = DECAY_VALUES[0]

final_model = fit_model(pre_test_df, best_decay)
final_metrics = evaluate(final_model, final_test_df)

print(f"Decay              : {best_decay}")
print(f"Fitted rho         : {final_model['rho']:+.4f}")
print()
print(f"Accuracy           : {final_metrics['accuracy']:.2%}")
print(f"Log loss           : {final_metrics['log_loss']:.4f}")
print(f"Home MAE           : {final_metrics['home_mae']:.4f}")
print(f"Away MAE           : {final_metrics['away_mae']:.4f}")
print(f"Total MAE          : {final_metrics['total_mae']:.4f}")
print(f"Mean actual score probability: {final_metrics['mean_actual_score_probability']:.4%}")
print()

final_results = final_metrics["results"]

print("Predictions:")
print(final_results["Prediction"].value_counts().sort_index())
print()
print("Actual results:")
print(final_results["FTR"].value_counts().sort_index())
print()

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print()
print("Static Poisson (no time decay)      : acc 47.63%  logloss 1.0494")
print("Time-decay Poisson (decay=0.003)    : acc 45.79%  logloss 1.0405")
print("Time-decay Poisson + DC             : acc 45.79%  logloss 1.0404")
print(f"Home/away Poisson + DC (this run)   : acc {final_metrics['accuracy']:.2%}  "
      f"logloss {final_metrics['log_loss']:.4f}")
print()

# ============================================================
# SAVE
# ============================================================

os.makedirs("data", exist_ok=True)

prediction_file = "data/home_away_poisson_2025_26.csv"
final_results.to_csv(prediction_file, index=False)

print(f"Saved predictions to: {prediction_file}")