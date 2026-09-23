import duckdb


database_path = "database/sports_ai.duckdb"

connection = duckdb.connect(database_path)

query = """
SELECT
    HomeTeam,
    COUNT(*) AS HomeMatches,
    SUM(CASE WHEN FTR = 'H' THEN 1 ELSE 0 END) AS HomeWins,
    SUM(CASE WHEN FTR = 'D' THEN 1 ELSE 0 END) AS Draws,
    SUM(CASE WHEN FTR = 'A' THEN 1 ELSE 0 END) AS HomeLosses
FROM premier_league_matches
GROUP BY HomeTeam
ORDER BY HomeWins DESC
"""

result = connection.execute(query).fetchdf()

print(result.to_string(index=False))

connection.close()