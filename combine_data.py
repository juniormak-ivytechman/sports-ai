from pathlib import Path

import pandas as pd


# ---------------------------------------
# SETTINGS
# ---------------------------------------

DATA_FOLDER = Path("data")

OUTPUT_FILE = (
    DATA_FOLDER / "premier_league_master.csv"
)


# ---------------------------------------
# FIND RAW SEASON FILES
# ---------------------------------------
files = [
    DATA_FOLDER / "premier_league_1819.csv",
    DATA_FOLDER / "premier_league_1920.csv",
    DATA_FOLDER / "premier_league_2021.csv",
    DATA_FOLDER / "premier_league_2122.csv",
    DATA_FOLDER / "premier_league_2223.csv",
    DATA_FOLDER / "premier_league_2324.csv",
    DATA_FOLDER / "premier_league_2425.csv",
    DATA_FOLDER / "premier_league_2526.csv",
    DATA_FOLDER / "premier_league_2627.csv",
]

print()
print("BUILDING CLEAN MASTER DATASET")
print("=" * 60)
print()


all_matches = []


# ---------------------------------------
# READ EACH SEASON
# ---------------------------------------

for file in files:

    print(f"Reading {file.name}...")

    # Keep Date as text so pandas cannot
    # guess the date format incorrectly.
    df = pd.read_csv(
        file,
        dtype={"Date": "string"}
    )

    # -----------------------------------
    # PARSE DATE EXPLICITLY
    # -----------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Check for parsing failures
    invalid_dates = df["Date"].isna().sum()

    if invalid_dates > 0:

        raise ValueError(
            f"{file.name} contains "
            f"{invalid_dates} invalid dates."
        )


    # -----------------------------------
    # CREATE SEASON COLUMN
    # -----------------------------------

    season = file.stem.replace(
        "premier_league_",
        ""
    )

    df["Season"] = season


    # -----------------------------------
    # STORE
    # -----------------------------------

    all_matches.append(df)


# ---------------------------------------
# COMBINE
# ---------------------------------------

master_df = pd.concat(
    all_matches,
    ignore_index=True
)


# ---------------------------------------
# SORT CHRONOLOGICALLY
# ---------------------------------------

master_df = master_df.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)


# ---------------------------------------
# CHECK DUPLICATES
# ---------------------------------------

duplicates = master_df.duplicated(
    subset=[
        "Date",
        "HomeTeam",
        "AwayTeam"
    ]
).sum()


if duplicates > 0:

    raise ValueError(
        f"Found {duplicates} duplicate matches."
    )


# ---------------------------------------
# SAVE DATE IN ISO FORMAT
# ---------------------------------------

master_df["Date"] = master_df[
    "Date"
].dt.strftime("%Y-%m-%d")


# ---------------------------------------
# SAVE
# ---------------------------------------

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------
# SUMMARY
# ---------------------------------------

print()
print("MASTER DATASET CREATED")
print("-" * 40)

print(
    "Total matches:",
    len(master_df)
)

print(
    "Duplicate matches:",
    duplicates
)

print(
    "First date:",
    master_df["Date"].min()
)

print(
    "Last date:",
    master_df["Date"].max()
)

print()

print("Matches by season:")

print(
    master_df["Season"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print(
    "Saved to:",
    OUTPUT_FILE
)