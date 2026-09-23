import duckdb


database_path = "database/sports_ai.duckdb"

connection = duckdb.connect(database_path)

result = connection.execute("""
    SELECT
        Date,
        HomeTeam,
        AwayTeam
    FROM premier_league_matches
    ORDER BY Date
    LIMIT 10
""").fetchdf()

print(result.to_string(index=False))

connection.close()