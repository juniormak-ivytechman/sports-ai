import requests


API_KEY = "123"
LEAGUE_ID = "4328"
SEASON = "2024-2025"

url = (
    f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"
    f"/eventsseason.php?id={LEAGUE_ID}&s={SEASON}"
)

response = requests.get(url)

print("Status code:", response.status_code)

data = response.json()

events = data.get("events", [])

print("Number of matches:", len(events))
print()

for event in events[:10]:
    print(
        event["dateEvent"],
        "|",
        event["strHomeTeam"],
        "vs",
        event["strAwayTeam"],
        "| Score:",
        event["intHomeScore"],
        "-",
        event["intAwayScore"]
    )