# sports-ai — Premier League Match Prediction

A full-stack football analytics and prediction project. Builds statistical
models of English Premier League matches from historical results and match
statistics, evaluates them rigorously against bookmaker closing odds,
produces calibrated probability estimates for all major betting markets,
and serves them through a live web interface with LLM-powered explanations.

**Live:** [sports-ai-bay.vercel.app](https://sports-ai-bay.vercel.app)

**Read this file first.** It's the canonical summary of the project.

---

## TL;DR — what this project actually does

- Predicts English Premier League match outcomes and betting markets
  from historical results and match statistics.
- Uses three independent modelling families: **Elo ratings**, **time-decay
  Dixon-Coles Poisson**, and **logistic regression** on engineered features.
- Benchmarks every model against the bookmaker closing line, using
  margin-normalized implied probabilities.
- Produces probabilities for **9 betting markets**: 1X2, Over 1.5, Over 2.5,
  Over 3.5, BTTS, Double Chance (1X, X2, 12), Draw No Bet, and Team to Score.
- Serves predictions through a **FastAPI backend** and a **Next.js frontend**,
  both deployed on free-tier hosting.
- Adds **Groq LLM explanations** that describe the model's reasoning in
  plain English, grounded strictly in the model's numbers.
- **Honest result: the model does not beat the market.** Football-only
  signals derived from public data are already priced into closing odds.
  See "Results" and "What we learned" below.

---

## Goal

Predict English Premier League match outcomes and betting markets using
statistical and ML models, benchmarked against bookmaker odds. This is a
portfolio-style quant modelling project: rigor, honest out-of-sample testing,
and defensible metrics matter more than chasing accuracy.

---

## Architecture
┌─────────────────────────┐
│ football-data.co.uk │ ← historical CSVs
│ football-data.org │ ← live fixtures API
└───────────┬─────────────┘
│
┌───────────▼─────────────┐
│ Python data pipeline │
│ combine / audit / │
│ feature engineering │
└───────────┬─────────────┘
│
┌───────────▼─────────────┐
│ Poisson + Dixon-Coles │ ← the model
│ (time-decay weighted) │
└───────────┬─────────────┘
│
┌───────────▼─────────────┐
│ FastAPI backend │ ← Render (free tier)
│ /predict /explain │
│ /fixtures/today /teams │
└───────────┬─────────────┘
│
┌───────────▼─────────────┐
│ Next.js frontend │ ← Vercel (free tier)
│ Today / Predict pages │
│ Dark mode + LLM card │
└─────────────────────────┘



The LLM is a **presentation layer, not a data layer.** The model produces
the numbers; the LLM only writes prose about those numbers. It never
invents statistics or predictions.

---

## Environment

- macOS, `~/Documents/sports-ai`, Python 3.13 virtual environment (`.venv`)
- Backend: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scipy`,
  `scikit-learn`, `groq`, `python-dotenv`
- Frontend: Next.js 16, Tailwind CSS v4, TypeScript, `axios`
- Data: DuckDB, football-data.co.uk CSVs, football-data.org API

---

## Repo layout
sports-ai/
├── data/ # raw season CSVs + master dataset
│ ├── premier_league_1819.csv ... premier_league_2627.csv
│ ├── premier_league_master.csv # canonical combined (3,090 matches)
│ └── premier_league_features.csv # rolling point-in-time features
├── database/
│ └── setup_database.py # DuckDB schema + load
├── backend/ # FastAPI service (deployed on Render)
│ ├── app/
│ │ ├── main.py # API routes
│ │ ├── model.py # Poisson + Dixon-Coles fit/predict
│ │ ├── football_api.py # football-data.org + API-Football
│ │ └── llm.py # Groq explanation layer
│ ├── requirements.txt
│ └── render.yaml
├── frontend/ # Next.js app (deployed on Vercel)
│ ├── app/
│ │ ├── layout.tsx # nav + dark mode bootstrap
│ │ ├── page.tsx # Today page
│ │ └── predict/page.tsx # Manual fixture predictor
│ ├── components/
│ │ ├── MarketCard.tsx # the market card UI
│ │ └── ThemeToggle.tsx
│ └── lib/api.ts # API client
├── src/
│ └── models/ # offline model experiments
│ ├── poisson_model.py # static Poisson baseline
│ ├── time_decay_poisson.py # time-decay + Dixon-Coles
│ ├── market_predictions.py # all 9 markets from score matrix
│ ├── simulate_live.py # holdout-of-N backtest
│ ├── ensemble*.py # ensemble experiments
│ └── value_dc_x2.py # value detection on DC X2
├── services/ # early API test scripts
└── README.md # this file


---

## Data pipeline

### Source

Nine seasons of Premier League data from `football-data.co.uk`, covering
2018/19 through 2026/27 (currently in progress). Each season is a single
CSV with results, match statistics (shots, shots on target, corners, cards),
and bookmaker odds.

### Canonical master file

`data/premier_league_master.csv` is built by `combine_data.py` from the nine
raw season files.

**Two facts every downstream script must get right** — both were the source
of real bugs:

- `Date` is stored as an **ISO string, `%Y-%m-%d`** (explicitly parsed from
  the raw `%d/%m/%Y` format during the master rebuild, then re-saved as ISO).
  Parsing it with any other format string silently produces `NaT`.
- `Season` is a **4-digit code taken from the filename**, e.g. `"2526"`,
  `"2627"` — **not** `"2025/26"`.

### Current master

- **3,090 matches**, 9 seasons
- **0 duplicates**
- **10 Aug 2018 – 20 Sep 2026**
- **100% coverage** on core result/statistics fields
- **~87.7% coverage** on bookmaker odds (`AvgH`, `AvgD`, `AvgA`)

---

## Modelling protocol

This split is used **consistently** across every offline experiment:

| Split | Date range | Purpose |
|---|---|---|
| **Development** | `Date < 2023-08-01` | Fit model parameters |
| **Validation** | `2023-08-01 <= Date < 2025-08-01` | Tune hyperparameters |
| **Final test** | `Date >= 2025-08-01` | Evaluated **exactly once**, never used for tuning |

**Hard rule:** never tune anything against the final test season. Enforced
throughout the offline experiments.

**Metrics:** accuracy for classification, log loss for probability quality,
Brier score for calibration. Market benchmark uses margin-normalized
`1/AvgH`, `1/AvgD`, `1/AvgA`.

**For the deployed backend**, the model simply fits on all available data
through today. No train/test split — the model is a live predictor, not
being evaluated.

---

## Models

### 1. Elo rating (offline experiment only)

Dynamic team strength. Starts every team at 1500. Updates with `K=40`,
home advantage `+40` (both tuned on validation). Converts Elo difference
to 1X2 probabilities via empirical band-counting.

Walk-forward home-win accuracy: **61–67%** across 2022/23 through 2025/26.

### 2. Time-decay Dixon-Coles Poisson (**deployed model**)

Per-team attack and defence parameters, fitted via maximum likelihood with:
- **Exponential time decay** (`ξ = 0.003`, tuned on validation).
- **Dixon-Coles low-score correction** (`ρ` fitted per model — near zero
  in practice for this dataset).
- **Regularisation** toward attack=1, defence=1.

Produces a full score matrix (0–0 through 9–9) from which **9 markets**
are derived. Home advantage: ~1.19.

### 3. Logistic regression on engineered features (offline experiment only)

21 features: rolling 5-match form points, home/away-specific form, rolling
goals scored/conceded, shots, shots on target, corners, plus corresponding
home-minus-away differences.

### 4. LLM explanation layer (deployed)

Groq's `openai/gpt-oss-120b` model writes 3–4 sentence plain-English
explanations of each prediction. The LLM receives structured facts from
the model and is instructed to never invent statistics. Fallback model
chain: `gpt-oss-120b` → `gpt-oss-20b` → `qwen/qwen3.6-27b`.

---

## Results — 2025/26 final test (380 matches, offline experiments)

### 1X2 accuracy and log loss

| System | Accuracy | Log loss |
|---|---:|---:|
| **Market** (bookmaker, margin-normalized) | 49.47% | **1.0153** |
| Equal-weight ensemble | 48.16% | 1.0294 |
| Weighted ensemble (0.75/0.20/0.05) | 49.21% | 1.0260 |
| Elo 1X2 | 50.26% | 1.0274 |
| Time-decay Poisson + DC | 45.79% | 1.0404 |
| Poisson baseline (static) | 47.63% | 1.0494 |
| Logistic regression (ML #4) | 49.47% | 1.0512 |

**The market is the strongest single probability estimator. This is not a
failure — it is the correct, expected result.**

### Market accuracy — derived from Poisson score matrix

| Market | Base rate | Model acc. | Verdict |
|---|---:|---:|---|
| Home Team to Score | 77.4% | 77.4% | tracks base rate |
| Over 1.5 goals | 78.9% | 78.9% | tracks base rate |
| Away Team to Score | 71.6% | 72.4% | tracks base rate |
| Double Chance 1X | 70.0% | 68.9% | slightly worse |
| Double Chance 12 | 72.6% | 72.6% | tracks base rate |
| Over 3.5 goals | 71.6% | 70.8% | slightly worse |
| Double Chance X2 | 57.4% | 64.7% | above base rate* |
| BTTS (Yes) | 56.1% | 57.6% | flat |
| Over 2.5 goals | 55.0% | 54.2% | flat |
| Draw No Bet (home) | 57.4% | 56.8% | slightly worse |
| 1X2 | 44.0% | 45.8% | flat |

*The apparent DC X2 edge was investigated further — see below.

### Value detection — the decisive test

For Double Chance X2, we tested whether the model beat the **market**:

| Threshold (model − market) | N picks | Model avg | Market avg | Actual | ROI |
|---:|---:|---:|---:|---:|---:|
| ≥ 2% | 187 | 0.574 | 0.489 | 0.471 | **−7.80%** |
| ≥ 5% | 132 | 0.579 | 0.475 | 0.485 | **−4.73%** |
| ≥ 10% | 52 | 0.594 | 0.438 | 0.423 | **−2.50%** |
| ≥ 15% | 24 | 0.621 | 0.422 | 0.292 | **−35.19%** |

**At every threshold, the market was closer to the actual outcome than the
model.** Conclusion: the model has no exploitable edge over the closing line.

---

## What we tried — and what worked or didn't

| Upgrade | Verdict |
|---|---|
| **Time decay** on Poisson | ✅ Real improvement (log loss 1.0494 → 1.0404) |
| **Dixon-Coles correction** | ➖ Flat (1.0405 → 1.0404). Not worth it. |
| **Home/away split** (4 params per team) | ❌ Regression on validation |
| **Equal-weight ensemble** | ✅ Small gain (1.0316 → 1.0294) |
| **Weighted ensemble** | ✅ Small additional gain (1.0294 → 1.0260) |
| **Adding market to blend** | ❌ Market takes 100% weight |
| **Value betting on DC X2** | ❌ Negative ROI at every threshold |
| **Elo 1X2 via polynomial mapping** | ❌ Overfit; worse than band-counting |
| **Elo 1X2 via band-counting** | ✅ Best single non-market model (1.0274) |

---

## What we learned

1. **Football-only modelling on public data has a ceiling.** Results and
   match statistics cannot close the gap to the closing line. That gap
   already contains information the model doesn't see: injuries, team
   news, xG, money flow, and every sharp bettor's opinion.

2. **Base rate is not the same as edge.** Markets like Over 1.5 have 78%
   base rates — predicting "over" every time gets 78%. The model matches,
   not beats, that number.

3. **Confidence-filtered accuracy is real but not exploitable.** Top-10%
   predictions hit 84–94% on several markets, but the market also assigns
   high confidence to those same matches.

4. **Calibration is more important than accuracy.** Log loss exposed the
   failures (zero-draw prediction, home-bias) that accuracy alone hid.

5. **The LLM cannot be a data layer.** It writes prose about numbers your
   model has already produced. It does not know fixtures, injuries, or
   current form. Feed it facts, or it invents them.

6. **The market is a hard benchmark, and that's fine.** A model that
   matches the market on football-only data is a legitimate result.

---

## Running locally

### Backend

```bash
cd ~/Documents/sports-ai/backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8000
Test:

bash
curl http://localhost:8000/health
curl "http://localhost:8000/predict?home=Arsenal&away=Chelsea"
curl "http://localhost:8000/explain?home=Arsenal&away=Chelsea"
Frontend

bash
cd ~/Documents/sports-ai/frontend
npm run dev
Open http://localhost:3000.

Rebuilding the master dataset

If you add a new season CSV to data/:


python combine_data.py
python audit_data.py
rm database/sports_ai.duckdb
python database/setup_database.py
The backend refits the model on every startup, so no manual retraining is
required. Just restart uvicorn.

Environment variables

Backend needs a .env at the repo root:

FOOTBALL_DATA_TOKEN=<32-char football-data.org token>
GROQ_API_KEY=gsk_<groq API key>
API_FOOTBALL_KEY=<api-football key>   # optional, injuries only
On Render, these are set via the dashboard's Environment tab — not via a
committed .env file.

Frontend needs one env var on Vercel:

NEXT_PUBLIC_API_URL=https://sports-ai-backend-dhr2.onrender.com
Deployment

Service	What	Plan	Notes
GitHub	Source of truth	Free	Private repo
Render	FastAPI backend	Free	Sleeps after 15 min idle — UptimeRobot keeps it warm
Vercel	Next.js frontend	Free	Auto-deploys on push
Groq	LLM	Free	gpt-oss-120b — 1,000 req/day
UptimeRobot	Backend pinger	Free	5-min interval prevents Render sleep
Auto-deploy flow: git push → Render and Vercel both detect the change
and redeploy. No manual intervention.

Known pitfalls

Do not paste large scripts into Terminal via cat > file <<'PY' ... PY.
Terminal.app chokes on large pastes. Use open -e with TextEdit.
Always state the Date format explicitly when parsing — never let
pandas guess. Silent NaT corruption is the classic failure mode.
Season filters use "2627"-style codes, not "2026/27".
Never tune a hyperparameter against the final test season.
AS is a SQL keyword. Quote it as "AS" in DuckDB queries.
The free tier of API-Football only covers seasons 2022–2024. Injury
data for the current season is not available. The code handles this
gracefully by returning empty injury lists.
Groq retires models without much notice. The LLM module includes a
fallback chain — if the first model 404s, it tries the next.