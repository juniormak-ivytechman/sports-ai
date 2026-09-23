import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/premier_league_master.csv"

STARTING_RATING = 1500

K_FACTOR = 40
HOME_ADVANTAGE = 40


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
# BUILD PRE-MATCH ELO DIFFERENCES
# ============================================================

ratings = {}


def get_rating(team):
    return ratings.get(
        team,
        STARTING_RATING
    )


def expected_home_probability(
    home_rating,
    away_rating
):

    adjusted_home = (
        home_rating
        + HOME_ADVANTAGE
    )

    return (
        1
        /
        (
            1
            +
            10
            **
            (
                (
                    away_rating
                    - adjusted_home
                )
                / 400
            )
        )
    )


rows = []


for _, match in df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_rating = get_rating(home_team)
    away_rating = get_rating(away_team)

    elo_difference = (
        home_rating
        + HOME_ADVANTAGE
        - away_rating
    )


    rows.append({
        "Date": match["Date"],
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTR": match["FTR"],
        "EloDifference": elo_difference
    })


    probability = expected_home_probability(
        home_rating,
        away_rating
    )


    # ----------------------------------------
    # ACTUAL RESULT
    # ----------------------------------------

    if match["FTR"] == "H":

        actual_home = 1.0
        actual_away = 0.0

    elif match["FTR"] == "D":

        actual_home = 0.5
        actual_away = 0.5

    else:

        actual_home = 0.0
        actual_away = 1.0


    # ----------------------------------------
    # UPDATE RATINGS AFTER MATCH
    # ----------------------------------------

    ratings[home_team] = (
        home_rating
        +
        K_FACTOR
        *
        (
            actual_home
            - probability
        )
    )

    ratings[away_team] = (
        away_rating
        +
        K_FACTOR
        *
        (
            actual_away
            - (1 - probability)
        )
    )


elo_df = pd.DataFrame(rows)


# ============================================================
# DEVELOPMENT / VALIDATION / FINAL TEST WINDOWS
# ============================================================

development_df = elo_df[
    elo_df["Date"] < "2023-08-01"
].copy()

validation_df = elo_df[
    (
        elo_df["Date"] >= "2023-08-01"
    )
    &
    (
        elo_df["Date"] < "2025-08-01"
    )
].copy()

final_test_df = elo_df[
    elo_df["Date"] >= "2025-08-01"
].copy()


# ============================================================
# TRAINING DATA
# ============================================================

X_development = development_df[
    ["EloDifference"]
]

y_development = development_df["FTR"]


X_validation = validation_df[
    ["EloDifference"]
]

y_validation = validation_df["FTR"]


# ============================================================
# MODEL SEARCH
# ============================================================

degrees = [1, 2, 3, 4]

C_values = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]


results = []


for degree in degrees:

    for C in C_values:

        model = Pipeline([
            (
                "polynomial",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=5000
                )
            )
        ])


        model.fit(
            X_development,
            y_development
        )


        validation_predictions = (
            model.predict(
                X_validation
            )
        )


        validation_probabilities = (
            model.predict_proba(
                X_validation
            )
        )


        accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )


        logloss = log_loss(
            y_validation,
            validation_probabilities,
            labels=model.named_steps[
                "classifier"
            ].classes_
        )


        results.append({
            "Degree": degree,
            "C": C,
            "Accuracy": accuracy,
            "LogLoss": logloss
        })


results_df = pd.DataFrame(results)


# ============================================================
# SORT BY LOG LOSS
# ============================================================

results_by_logloss = (
    results_df
    .sort_values(
        "LogLoss",
        ascending=True
    )
)


# ============================================================
# OUTPUT VALIDATION RESULTS
# ============================================================

print()
print("ELO 1X2 PROBABILITY MODEL SEARCH")
print("=" * 70)

print()

print("Development period:")
print(
    development_df["Date"].min().date(),
    "to",
    development_df["Date"].max().date()
)

print()

print("Validation period:")
print(
    validation_df["Date"].min().date(),
    "to",
    validation_df["Date"].max().date()
)

print()

print("Parameter results:")
print()

print(
    results_by_logloss.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2%}".format,
            "LogLoss": "{:.4f}".format
        }
    )
)


# ============================================================
# SELECT BEST MODEL BY LOG LOSS
# ============================================================

best = results_by_logloss.iloc[0]

best_degree = int(best["Degree"])
best_C = float(best["C"])


print()

print("BEST MODEL")
print("-" * 40)

print(
    "Polynomial degree:",
    best_degree
)

print(
    "C:",
    best_C
)

print(
    f"Validation accuracy: "
    f"{best['Accuracy']:.2%}"
)

print(
    f"Validation log loss: "
    f"{best['LogLoss']:.4f}"
)


# ============================================================
# REFIT ON ALL PRE-TEST DATA
# ============================================================

pre_test_df = elo_df[
    elo_df["Date"] < "2025-08-01"
].copy()

X_pre_test = pre_test_df[
    ["EloDifference"]
]

y_pre_test = pre_test_df["FTR"]


final_model = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=best_degree,
            include_bias=False
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            C=best_C,
            max_iter=5000
        )
    )
])


final_model.fit(
    X_pre_test,
    y_pre_test
)


# ============================================================
# FINAL 2025/26 TEST
# ============================================================

X_final_test = final_test_df[
    ["EloDifference"]
]

y_final_test = final_test_df["FTR"]


final_predictions = final_model.predict(
    X_final_test
)

final_probabilities = (
    final_model.predict_proba(
        X_final_test
    )
)


final_accuracy = accuracy_score(
    y_final_test,
    final_predictions
)

final_logloss = log_loss(
    y_final_test,
    final_probabilities,
    labels=final_model.named_steps[
        "classifier"
    ].classes_
)


# ============================================================
# PROBABILITY TABLE
# ============================================================

classes = (
    final_model
    .named_steps[
        "classifier"
    ]
    .classes_
)


probability_lookup = {
    class_name: index
    for index, class_name
    in enumerate(classes)
}


final_results = final_test_df[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "EloDifference"
    ]
].reset_index(drop=True)


final_results["HomeProbability"] = (
    final_probabilities[
        :,
        probability_lookup["H"]
    ]
)

final_results["DrawProbability"] = (
    final_probabilities[
        :,
        probability_lookup["D"]
    ]
)

final_results["AwayProbability"] = (
    final_probabilities[
        :,
        probability_lookup["A"]
    ]
)

final_results["Prediction"] = (
    final_predictions
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("FINAL 2025/26 TEST")
print("=" * 70)

print()

print(
    "Accuracy:",
    f"{final_accuracy:.2%}"
)

print(
    "Log loss:",
    f"{final_logloss:.4f}"
)

print()

print("Predictions:")

print(
    pd.Series(
        final_predictions
    )
    .value_counts()
    .sort_index()
)

print()

print("Actual results:")

print(
    y_final_test
    .value_counts()
    .sort_index()
)

print()

print("First 15 predictions:")

print()

print(
    final_results
    [
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "HomeProbability",
            "DrawProbability",
            "AwayProbability",
            "Prediction"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/elo_1x2_tuned_2025_26.csv"
)

final_results.to_csv(
    output_file,
    index=False
)

print()

print(
    "Saved to:",
    output_file
)