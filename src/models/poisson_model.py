import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import poisson


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/premier_league_master.csv"

STARTING_ATTACK = 1.0
STARTING_DEFENCE = 1.0

HOME_ADVANTAGE = 1.10

DECAY_DAYS = 365


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


# ============================================================
# PREPARE HISTORICAL WINDOW
# ============================================================

TRAIN_END = pd.Timestamp(
    "2025-08-01"
)

train_df = df[
    df["Date"] < TRAIN_END
].copy()


# ============================================================
# CREATE TEAM INDEX
# ============================================================

teams = sorted(
    set(train_df["HomeTeam"])
    |
    set(train_df["AwayTeam"])
)

team_to_index = {
    team: index
    for index, team
    in enumerate(teams)
}

n_teams = len(teams)


# ============================================================
# INITIAL PARAMETERS
# ============================================================

# We use:
#
# attack[team]
# defence[team]
#
# plus a home-advantage parameter.
#
# A constraint is imposed so that attack and defence
# parameters remain identifiable.

initial_attack = np.ones(
    n_teams
) * STARTING_ATTACK

initial_defence = np.ones(
    n_teams
) * STARTING_DEFENCE

initial_home_advantage = np.log(
    HOME_ADVANTAGE
)


initial_parameters = np.concatenate([
    initial_attack,
    initial_defence,
    [initial_home_advantage]
])


# ============================================================
# LOG LIKELIHOOD
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

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        home_index = team_to_index[
            home_team
        ]

        away_index = team_to_index[
            away_team
        ]


        # Expected goals

        home_lambda = (
            home_advantage
            *
            attack[home_index]
            *
            defence[away_index]
        )

        away_lambda = (
            attack[away_index]
            *
            defence[home_index]
        )


        # Prevent invalid values

        home_lambda = max(
            home_lambda,
            0.01
        )

        away_lambda = max(
            away_lambda,
            0.01
        )


        home_goals = int(
            match["FTHG"]
        )

        away_goals = int(
            match["FTAG"]
        )


        # Add Poisson log probability

        log_likelihood += poisson.logpmf(
            home_goals,
            home_lambda
        )

        log_likelihood += poisson.logpmf(
            away_goals,
            away_lambda
        )


    # Regularisation:
    #
    # Prevent extreme attack/defence values.

    regularisation = (
        0.01
        *
        (
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


# ============================================================
# CHECK OPTIMISATION
# ============================================================

print()

print(
    "POISSON MODEL FIT"
)

print("=" * 60)

print()

print(
    "Optimisation successful:",
    result.success
)

print(
    "Message:",
    result.message
)

print(
    "Training matches:",
    len(train_df)
)


# ============================================================
# EXTRACT PARAMETERS
# ============================================================

fitted_parameters = result.x

attack = fitted_parameters[
    :n_teams
]

defence = fitted_parameters[
    n_teams:2 * n_teams
]

home_advantage = np.exp(
    fitted_parameters[-1]
)


# ============================================================
# TEAM RATINGS
# ============================================================

team_ratings = pd.DataFrame({
    "Team": teams,
    "Attack": attack,
    "Defence": defence
})


team_ratings["Attack"] = (
    team_ratings["Attack"]
    .round(3)
)

team_ratings["Defence"] = (
    team_ratings["Defence"]
    .round(3)
)


# ============================================================
# OUTPUT
# ============================================================

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
    "Strongest attacks:"
)

print(
    team_ratings
    .sort_values(
        "Attack",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print()

print(
    "Strongest defences:"
)

print(
    team_ratings
    .sort_values(
        "Defence"
    )
    .head(10)
    .to_string(index=False)
)