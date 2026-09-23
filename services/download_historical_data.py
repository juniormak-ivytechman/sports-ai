import requests
from pathlib import Path


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

SEASONS = [
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


data_folder = Path("data")

for season in SEASONS:
    url = BASE_URL.format(season=season)
    file_path = data_folder / f"premier_league_{season}.csv"

    print(f"Downloading {season}...")

    response = requests.get(url)

    if response.status_code == 200:
        file_path.write_bytes(response.content)
        print(f"Saved: {file_path}")
    else:
        print(f"Failed: HTTP {response.status_code}")

    print()