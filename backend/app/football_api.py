import os
import requests
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env before reading the token.
# The path is relative to this file, so it works regardless of cwd.
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()

# Debug line — remove later if you want
print(f"[football_api] Token loaded: {'YES (' + str(len(FOOTBALL_DATA_TOKEN)) + ' chars)' if FOOTBALL_DATA_TOKEN else 'NO'}")

BASE_URL = "https://api.football-data.org/v4"

# Premier League competition ID on football-data.org
EPL_ID = 2021


def get_todays_matches():
    """Fetch today's EPL fixtures. Returns [] if no token or no matches."""
    if not FOOTBALL_DATA_TOKEN:
        print("[football_api] No FOOTBALL_DATA_TOKEN set. Returning empty list.")
        return []

    today = date.today().isoformat()
    url = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    params = {"dateFrom": today, "dateTo": today}
    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[football_api] Error fetching fixtures: {e}")
        return []

    matches = []
    for m in data.get("matches", []):
        matches.append({
            "id": m["id"],
            "utcDate": m["utcDate"],
            "homeTeam": m["homeTeam"]["name"],
            "awayTeam": m["awayTeam"]["name"],
            "status": m["status"],
        })
    return matches


def get_recent_matches(days_back=7):
    """Fetch last N days of EPL results (for updating team form)."""
    if not FOOTBALL_DATA_TOKEN:
        return []

    end = date.today()
    start = end - timedelta(days=days_back)
    url = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    params = {
        "dateFrom": start.isoformat(),
        "dateTo": end.isoformat(),
        "status": "FINISHED",
    }
    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception as e:
        print(f"[football_api] Error fetching recent matches: {e}")
        return []

def get_recent_matchday(days_back=14):
    """Find the most recent day with EPL fixtures within the last N days."""
    if not FOOTBALL_DATA_TOKEN:
        return []

    end = date.today()
    start = end - timedelta(days=days_back)
    url = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    params = {
        "dateFrom": start.isoformat(),
        "dateTo": end.isoformat(),
        "status": "FINISHED",
    }
    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        matches = r.json().get("matches", [])
    except Exception as e:
        print(f"[football_api] Error fetching recent matchday: {e}")
        return []

    if not matches:
        return []

    # Find the most recent date that has fixtures
    latest_date = max(m["utcDate"][:10] for m in matches)
    latest_matches = [m for m in matches if m["utcDate"].startswith(latest_date)]

    return [{
        "id": m["id"],
        "utcDate": m["utcDate"],
        "homeTeam": m["homeTeam"]["name"],
        "awayTeam": m["awayTeam"]["name"],
        "status": m["status"],
    } for m in latest_matches]