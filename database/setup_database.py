import duckdb


database_path = "database/sports_ai.duckdb"

connection = duckdb.connect(database_path)

connection.execute("""
    CREATE OR REPLACE TABLE premier_league_matches AS
    SELECT *
    FROM read_csv_auto('data/premier_league_master.csv')
""")

result = connection.execute("""
    SELECT COUNT(*)
    FROM premier_league_matches
""").fetchone()

print("Database created successfully.")
print("Matches stored:", result[0])

connection.close()