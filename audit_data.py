import pandas as pd


# ---------------------------------------
# LOAD
# ---------------------------------------

file_path = (
    "data/premier_league_master.csv"
)

df = pd.read_csv(file_path)


# ---------------------------------------
# PARSE DATE
# ---------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d",
    errors="raise"
)


print()
print("=" * 70)
print("SPORTS AI DATA AUDIT")
print("=" * 70)

print()


# ---------------------------------------
# BASIC
# ---------------------------------------

print("Rows:", len(df))
print("Columns:", len(df.columns))

print()


# ---------------------------------------
# DATE
# ---------------------------------------

print("DATE RANGE")
print("-" * 30)

print(
    "First:",
    df["Date"].min()
)

print(
    "Last: ",
    df["Date"].max()
)

print()


# ---------------------------------------
# DUPLICATES
# ---------------------------------------

duplicate_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam"
]

duplicates = df.duplicated(
    subset=duplicate_columns
).sum()

print("DUPLICATE MATCHES")
print("-" * 30)

print(
    "Duplicates:",
    duplicates
)

print()


# ---------------------------------------
# MISSING CORE DATA
# ---------------------------------------

important_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "AvgH",
    "AvgD",
    "AvgA"
]

print("MISSING VALUES")
print("-" * 30)

missing = (
    df[important_columns]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    missing.to_string()
)

print()


# ---------------------------------------
# RESULT DISTRIBUTION
# ---------------------------------------

print("RESULT DISTRIBUTION")
print("-" * 30)

print(
    df["FTR"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()


# ---------------------------------------
# SEASON COUNTS
# ---------------------------------------

print("SEASON COUNTS")
print("-" * 30)

print(
    df["Season"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()


# ---------------------------------------
# TEAM COUNT
# ---------------------------------------

teams = sorted(
    set(df["HomeTeam"])
    .union(
        set(df["AwayTeam"])
    )
)

print("TEAM INFORMATION")
print("-" * 30)

print(
    "Unique team names:",
    len(teams)
)

print()


# ---------------------------------------
# ODDS COVERAGE
# ---------------------------------------

print("ODDS COVERAGE")
print("-" * 30)

for column in [
    "AvgH",
    "AvgD",
    "AvgA"
]:

    available = (
        df[column]
        .notna()
        .sum()
    )

    percentage = (
        available
        / len(df)
        * 100
    )

    print(
        f"{column}: "
        f"{available} / {len(df)} "
        f"({percentage:.2f}%)"
    )

print()


# ---------------------------------------
# MATCH STATISTICS
# ---------------------------------------

print(
    "MATCH STATISTICS COVERAGE"
)

print("-" * 30)

for column in [
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC"
]:

    available = (
        df[column]
        .notna()
        .sum()
    )

    percentage = (
        available
        / len(df)
        * 100
    )

    print(
        f"{column}: "
        f"{available} / {len(df)} "
        f"({percentage:.2f}%)"
    )

print()


# ---------------------------------------
# CHRONOLOGY
# ---------------------------------------

print(
    "CHRONOLOGICAL ORDER"
)

print("-" * 30)

print(
    "Sorted correctly:",
    df["Date"]
    .is_monotonic_increasing
)

print()


# ---------------------------------------
# GOAL SANITY
# ---------------------------------------

print(
    "GOAL SANITY CHECK"
)

print("-" * 30)

print(
    "Invalid home goals:",
    (df["FTHG"] < 0).sum()
)

print(
    "Invalid away goals:",
    (df["FTAG"] < 0).sum()
)

print()


# ---------------------------------------
# FINAL
# ---------------------------------------

print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)