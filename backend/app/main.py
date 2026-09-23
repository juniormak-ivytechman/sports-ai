import os
from datetime import date
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.model import PoissonModel
from app.football_api import get_todays_matches, get_recent_matchday

load_dotenv(dotenv_path="../.env")

app = FastAPI(title="Sports AI", version="1.0.0")

# Allow the Next.js dev server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fit the model once at startup
print("[startup] Fitting Poisson model...")
model = PoissonModel()
model.fit()
print("[startup] Model ready.")


@app.get("/health")
def health():
    return {"status": "ok", "date": date.today().isoformat()}


@app.get("/fixtures/today")
def fixtures_today():
    """Today's EPL fixtures. Falls back to most recent matchday if none today."""
    matches = get_todays_matches()
    fallback = False

    if not matches:
        matches = get_recent_matchday(days_back=14)
        fallback = True

    if not matches:
        return {
            "date": date.today().isoformat(),
            "matches": [],
            "note": "No fixtures today and no recent matchday found in last 14 days.",
        }

    results = []
    for m in matches:
        pred = model.predict(m["homeTeam"], m["awayTeam"])
        if pred is None:
            pred = {"error": "Teams not in model"}
        results.append({
            "id": m["id"],
            "utcDate": m["utcDate"],
            "status": m["status"],
            "prediction": pred,
        })

    return {
        "date": date.today().isoformat(),
        "matches": results,
        "note": (
            "Showing most recent matchday (no fixtures today)."
            if fallback else None
        ),
    }


@app.get("/predict")
def predict(home: str, away: str):
    """Predict a specific fixture by team names."""
    pred = model.predict(home, away)
    if pred is None:
        raise HTTPException(status_code=404, detail="Teams not recognised")
    return pred


@app.get("/teams")
def teams():
    return {"teams": sorted(model.t2i.keys())}