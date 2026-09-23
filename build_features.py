import duckdb
import pandas as pd


DATABASE_PATH = "database/sports_ai.duckdb"

connection = duckdb.connect(DATABASE_PATH)

df = connection.execute("""
    SELECT
        Date,
        HomeTeam,
        AwayTeam,
        FTHG,
        FTAG,
        FTR,
        HS,
        "AS",
        HST,
        AST,
        HC,
        AC,
        AvgH,
        AvgD,
        AvgA
    FROM premier_league_matches
    ORDER BY Date
""").fetchdf()

connection.close()


df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------
# FEATURE STORAGE
# ---------------------------------------

home_form_points = []
away_form_points = []

home_team_home_form_points = []
away_team_away_form_points = []

home_avg_goals_scored = []
away_avg_goals_scored = []

home_avg_goals_conceded = []
away_avg_goals_conceded = []

home_avg_shots = []
away_avg_shots = []

home_avg_shots_on_target = []
away_avg_shots_on_target = []

home_avg_corners = []
away_avg_corners = []


# ---------------------------------------
# TEAM HISTORY
# ---------------------------------------

team_history = {}
home_history = {}
away_history = {}


# ---------------------------------------
# PROCESS MATCHES CHRONOLOGICALLY
# ---------------------------------------

for _, match in df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]


    # -----------------------------------
    # PREVIOUS MATCHES
    # -----------------------------------

    home_previous_5 = team_history.get(home_team, [])[-5:]
    away_previous_5 = team_history.get(away_team, [])[-5:]

    home_home_previous_5 = home_history.get(home_team, [])[-5:]
    away_away_previous_5 = away_history.get(away_team, [])[-5:]


    # -----------------------------------
    # FORM POINTS
    # -----------------------------------

    home_points = sum(
        x["points"] for x in home_previous_5
    )

    away_points = sum(
        x["points"] for x in away_previous_5
    )

    home_form_points.append(home_points)
    away_form_points.append(away_points)


    # -----------------------------------
    # HOME / AWAY FORM
    # -----------------------------------

    home_home_points = sum(
        x["points"] for x in home_home_previous_5
    )

    away_away_points = sum(
        x["points"] for x in away_away_previous_5
    )

    home_team_home_form_points.append(home_home_points)
    away_team_away_form_points.append(away_away_points)


    # -----------------------------------
    # GOALS
    # -----------------------------------

    if home_previous_5:

        home_scored_avg = sum(
            x["goals_scored"]
            for x in home_previous_5
        ) / len(home_previous_5)

        home_conceded_avg = sum(
            x["goals_conceded"]
            for x in home_previous_5
        ) / len(home_previous_5)

    else:

        home_scored_avg = 0
        home_conceded_avg = 0


    if away_previous_5:

        away_scored_avg = sum(
            x["goals_scored"]
            for x in away_previous_5
        ) / len(away_previous_5)

        away_conceded_avg = sum(
            x["goals_conceded"]
            for x in away_previous_5
        ) / len(away_previous_5)

    else:

        away_scored_avg = 0
        away_conceded_avg = 0


    home_avg_goals_scored.append(home_scored_avg)
    away_avg_goals_scored.append(away_scored_avg)

    home_avg_goals_conceded.append(home_conceded_avg)
    away_avg_goals_conceded.append(away_conceded_avg)


    # -----------------------------------
    # SHOTS
    # -----------------------------------

    if home_previous_5:

        home_shots_avg = sum(
            x["shots"]
            for x in home_previous_5
            if x["shots"] is not None
        ) / len([
            x for x in home_previous_5
            if x["shots"] is not None
        ])

        home_shots_on_target_avg = sum(
            x["shots_on_target"]
            for x in home_previous_5
            if x["shots_on_target"] is not None
        ) / len([
            x for x in home_previous_5
            if x["shots_on_target"] is not None
        ])

        home_corners_avg = sum(
            x["corners"]
            for x in home_previous_5
            if x["corners"] is not None
        ) / len([
            x for x in home_previous_5
            if x["corners"] is not None
        ])

    else:

        home_shots_avg = 0
        home_shots_on_target_avg = 0
        home_corners_avg = 0


    if away_previous_5:

        away_shots_avg = sum(
            x["shots"]
            for x in away_previous_5
            if x["shots"] is not None
        ) / len([
            x for x in away_previous_5
            if x["shots"] is not None
        ])

        away_shots_on_target_avg = sum(
            x["shots_on_target"]
            for x in away_previous_5
            if x["shots_on_target"] is not None
        ) / len([
            x for x in away_previous_5
            if x["shots_on_target"] is not None
        ])

        away_corners_avg = sum(
            x["corners"]
            for x in away_previous_5
            if x["corners"] is not None
        ) / len([
            x for x in away_previous_5
            if x["corners"] is not None
        ])

    else:

        away_shots_avg = 0
        away_shots_on_target_avg = 0
        away_corners_avg = 0


    home_avg_shots.append(home_shots_avg)
    away_avg_shots.append(away_shots_avg)

    home_avg_shots_on_target.append(home_shots_on_target_avg)
    away_avg_shots_on_target.append(away_shots_on_target_avg)

    home_avg_corners.append(home_corners_avg)
    away_avg_corners.append(away_corners_avg)


    # -----------------------------------
    # ACTUAL RESULT
    # -----------------------------------

    if match["FTR"] == "H":

        home_result = 3
        away_result = 0

    elif match["FTR"] == "D":

        home_result = 1
        away_result = 1

    else:

        home_result = 0
        away_result = 3


    # -----------------------------------
    # STORE CURRENT MATCH IN HISTORY
    # -----------------------------------

    home_match_data = {
        "points": home_result,
        "goals_scored": match["FTHG"],
        "goals_conceded": match["FTAG"],
        "shots": match["HS"],
        "shots_on_target": match["HST"],
        "corners": match["HC"]
    }

    away_match_data = {
        "points": away_result,
        "goals_scored": match["FTAG"],
        "goals_conceded": match["FTHG"],
        "shots": match["AS"],
        "shots_on_target": match["AST"],
        "corners": match["AC"]
    }


    team_history.setdefault(
        home_team, []
    ).append(home_match_data)

    team_history.setdefault(
        away_team, []
    ).append(away_match_data)


    home_history.setdefault(
        home_team, []
    ).append(home_match_data)

    away_history.setdefault(
        away_team, []
    ).append(away_match_data)


# ---------------------------------------
# ADD FEATURES
# ---------------------------------------

df["HomeFormPoints"] = home_form_points
df["AwayFormPoints"] = away_form_points

df["HomeTeamHomeFormPoints"] = home_team_home_form_points
df["AwayTeamAwayFormPoints"] = away_team_away_form_points

df["HomeAvgGoalsScored"] = home_avg_goals_scored
df["AwayAvgGoalsScored"] = away_avg_goals_scored

df["HomeAvgGoalsConceded"] = home_avg_goals_conceded
df["AwayAvgGoalsConceded"] = away_avg_goals_conceded

df["HomeAvgShots"] = home_avg_shots
df["AwayAvgShots"] = away_avg_shots

df["HomeAvgShotsOnTarget"] = home_avg_shots_on_target
df["AwayAvgShotsOnTarget"] = away_avg_shots_on_target

df["HomeAvgCorners"] = home_avg_corners
df["AwayAvgCorners"] = away_avg_corners


# ---------------------------------------
# SAVE
# ---------------------------------------

output_file = "data/premier_league_features.csv"

df.to_csv(output_file, index=False)


print("Feature dataset created.")
print("Total matches:", len(df))
print("Saved to:", output_file)

print()

print(
    df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeFormPoints",
            "AwayFormPoints",
            "HomeAvgGoalsScored",
            "AwayAvgGoalsScored",
            "HomeAvgShots",
            "AwayAvgShots",
            "HomeAvgShotsOnTarget",
            "AwayAvgShotsOnTarget",
            "HomeAvgCorners",
            "AwayAvgCorners"
        ]
    ]
    .head(20)
    .to_string(index=False)
)