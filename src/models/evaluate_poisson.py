import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import poisson

from sklearn.metrics import accuracy_score, log_loss


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/premier_league_master.csv"

TRAIN_END = "2025-08-01"

STARTING_ATTACK = 1.0
STARTING_DEFENCE = 1.0

HOME_ADVANTAGE = 1.10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d"
)

df = df.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)


train_df = df[
    df["Date"] < TRAIN_END
].copy()

test_df = df[
    df["Date"] >= TRAIN_END
].copy()


# ============================================================
# TEAM INDEX
# ============================================================

teams = sorted(
    set(train_df["HomeTeam"])
    |
    set(train_df["AwayTeam"])
)

team_to_index = {
    team: i
    for i, team in enumerate(teams)
}

n_teams = len(teams)


# ============================================================
# INITIAL PARAMETERS
# ============================================================

initial_attack = np.ones(
    n_teams
) * STARTING_ATTACK

initial_defence = np.ones(
    n_teams
) * STARTING_DEFENCE

initial_parameters = np.concatenate([
    initial_attack,
    initial_defence,
    [np.log(HOME_ADVANTAGE)]
])


# ============================================================
# LIKELIHOOD
# ============================================================

def negative_log_likelihood(
    parameters,
    matches
):

    attack = parameters[
        :n_teams
    ]

    defence = parameters[
        n_teams:2 * n_teams
    ]

    home_advantage = np.exp(
        parameters[-1]
    )

    log_likelihood = 0.0


    for _, match in matches.iterrows():

        home_index = team_to_index[
            match["HomeTeam"]
        ]

        away_index = team_to_index[
            match["AwayTeam"]
        ]


        home_lambda = (
            home_advantage
            * attack[home_index]
            * defence[away_index]
        )

        away_lambda = (
            attack[away_index]
            * defence[home_index]
        )


        home_lambda = max(
            home_lambda,
            0.01
        )

        away_lambda = max(
            away_lambda,
            0.01
        )


        log_likelihood += poisson.logpmf(
            int(match["FTHG"]),
            home_lambda
        )

        log_likelihood += poisson.logpmf(
            int(match["FTAG"]),
            away_lambda
        )


    regularisation = (
        0.01
        * (
            np.sum(
                (attack - 1.0) ** 2
            )
            +
            np.sum(
                (defence - 1.0) ** 2
            )
        )
    )


    return (
        -log_likelihood
        + regularisation
    )


# ============================================================
# FIT MODEL
# ============================================================

result = minimize(
    negative_log_likelihood,
    initial_parameters,
    args=(train_df,),
    method="L-BFGS-B",
    bounds=(
        [(0.05, 5.0)] * n_teams
        +
        [(0.05, 5.0)] * n_teams
        +
        [(-2.0, 2.0)]
    ),
    options={
        "maxiter": 1000
    }
)


if not result.success:

    raise RuntimeError(
        f"Poisson optimisation failed: "
        f"{result.message}"
    )


parameters = result.x

attack = parameters[
    :n_teams
]

defence = parameters[
    n_teams:2 * n_teams
]

home_advantage = np.exp(
    parameters[-1]
)


# ============================================================
# SCORE MATRIX
# ============================================================

def get_score_matrix(
    home_lambda,
    away_lambda,
    max_goals=10
):

    goals = np.arange(
        max_goals + 1
    )

    home_probs = poisson.pmf(
        goals,
        home_lambda
    )

    away_probs = poisson.pmf(
        goals,
        away_lambda
    )

    matrix = np.outer(
        home_probs,
        away_probs
    )

    matrix = (
        matrix
        /
        matrix.sum()
    )

    return matrix


# ============================================================
# PREDICT 2025/26
# ============================================================

rows = []


for _, match in test_df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]


    # --------------------------------------------------------
    # UNKNOWN TEAM HANDLING
    # --------------------------------------------------------
    #
    # If a team was absent from the training period,
    # use league-neutral attack and defence.
    #
    # This allows us to evaluate every test match instead
    # of silently deleting matches involving promoted teams.
    #

    if home_team in team_to_index:

        home_attack = attack[
            team_to_index[home_team]
        ]

        home_defence = defence[
            team_to_index[home_team]
        ]

    else:

        home_attack = 1.0
        home_defence = 1.0


    if away_team in team_to_index:

        away_attack = attack[
            team_to_index[away_team]
        ]

        away_defence = defence[
            team_to_index[away_team]
        ]

    else:

        away_attack = 1.0
        away_defence = 1.0


    # --------------------------------------------------------
    # EXPECTED GOALS
    # --------------------------------------------------------

    home_lambda = (
        home_advantage
        * home_attack
        * away_defence
    )

    away_lambda = (
        away_attack
        * home_defence
    )


    home_lambda = max(
        home_lambda,
        0.01
    )

    away_lambda = max(
        away_lambda,
        0.01
    )


    # --------------------------------------------------------
    # SCORELINE DISTRIBUTION
    # --------------------------------------------------------

    matrix = get_score_matrix(
        home_lambda,
        away_lambda
    )


    home_probability = 0.0
    draw_probability = 0.0
    away_probability = 0.0


    for home_goals in range(
        matrix.shape[0]
    ):

        for away_goals in range(
            matrix.shape[1]
        ):

            probability = matrix[
                home_goals,
                away_goals
            ]


            if home_goals > away_goals:

                home_probability += (
                    probability
                )

            elif home_goals == away_goals:

                draw_probability += (
                    probability
                )

            else:

                away_probability += (
                    probability
                )


    # --------------------------------------------------------
    # MOST LIKELY OUTCOME
    # --------------------------------------------------------

    probabilities = {
        "H": home_probability,
        "D": draw_probability,
        "A": away_probability
    }


    prediction = max(
        probabilities,
        key=probabilities.get
    )


    # --------------------------------------------------------
    # ACTUAL SCORE
    # --------------------------------------------------------

    actual_home_goals = int(
        match["FTHG"]
    )

    actual_away_goals = int(
        match["FTAG"]
    )


    actual_score_probability = (
        matrix[
            actual_home_goals,
            actual_away_goals
        ]
        if (
            actual_home_goals
            < matrix.shape[0]
            and
            actual_away_goals
            < matrix.shape[1]
        )
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
        "UnknownHomeTeam": home_team not in team_to_index,
        "UnknownAwayTeam": away_team not in team_to_index
    })


results_df = pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    results_df["FTR"],
    results_df["Prediction"]
)


probability_matrix = results_df[
    [
        "AwayProbability",
        "DrawProbability",
        "HomeProbability"
    ]
].values


model_log_loss = log_loss(
    results_df["FTR"],
    probability_matrix,
    labels=["A", "D", "H"]
)


home_goal_mae = (
    results_df["ExpectedHomeGoals"]
    -
    results_df["HomeGoals"]
).abs().mean()


away_goal_mae = (
    results_df["ExpectedAwayGoals"]
    -
    results_df["AwayGoals"]
).abs().mean()


total_goal_mae = (
    (
        results_df["ExpectedHomeGoals"]
        +
        results_df["ExpectedAwayGoals"]
    )
    -
    (
        results_df["HomeGoals"]
        +
        results_df["AwayGoals"]
    )
).abs().mean()


mean_actual_score_probability = (
    results_df["ActualScoreProbability"]
    .mean()
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("BASELINE POISSON EVALUATION")
print("=" * 70)

print()

print(
    "Training matches:",
    len(train_df)
)

print(
    "Testing matches:",
    len(results_df)
)

print()

print(
    "Unknown home-team predictions:",
    results_df["UnknownHomeTeam"].sum()
)

print(
    "Unknown away-team predictions:",
    results_df["UnknownAwayTeam"].sum()
)

print()

print(
    "Estimated home advantage:",
    round(
        home_advantage,
        3
    )
)

print()

print(
    f"1X2 accuracy: {accuracy:.2%}"
)

print(
    f"1X2 log loss: {model_log_loss:.4f}"
)

print()

print(
    f"Home-goal MAE: {home_goal_mae:.4f}"
)

print(
    f"Away-goal MAE: {away_goal_mae:.4f}"
)

print(
    f"Total-goal MAE: {total_goal_mae:.4f}"
)

print()

print(
    "Mean probability assigned "
    "to actual score:",
    f"{mean_actual_score_probability:.4%}"
)

print()

print("Predictions:")

print(
    results_df["Prediction"]
    .value_counts()
    .sort_index()
)

print()

print("Actual results:")

print(
    results_df["FTR"]
    .value_counts()
    .sort_index()
)

print()

print("First 15 predictions:")

print()

print(
    results_df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "ExpectedHomeGoals",
            "ExpectedAwayGoals",
            "HomeProbability",
            "DrawProbability",
            "AwayProbability",
            "Prediction",
            "ActualScoreProbability"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/poisson_baseline_2025_26.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()

print(
    "Saved to:",
    output_file
)
