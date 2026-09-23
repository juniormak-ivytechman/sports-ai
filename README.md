# sports-ai — Premier League Match Prediction

A full-stack football analytics and prediction project. Builds statistical
models of English Premier League matches from historical results and match
statistics, evaluates them rigorously against bookmaker closing odds, and
produces calibrated probability estimates for all major betting markets.

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

## Environment

- macOS, `~/Documents/sports-ai`, Python 3.13 virtual environment (`.venv`)
- Libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `duckdb`, `requests`
- No git repo initialized yet (see "Suggested next steps")

---

## Repo layout
sports-ai/
├── data/ # raw season CSVs + master + model outputs
│ ├── premier_league_1819.csv # raw, football-data.co.uk format
│ ├── ...
│ ├── premier_league_2526.csv
│ ├── premier_league_master.csv # combined, canonical (3,040 matches)
│ ├── premier_league_features.csv # rolling point-in-time features
│ ├── premier_league_elo.csv # Elo ratings per match
│ ├── market_predictions_2025_26.csv
│ ├── ensemble_weighted_2025_26.csv
│ └── ...
├── database/
│ ├── setup_database.py
│ └── sports_ai.duckdb
├── src/
│ ├── data/ # (scaffold, unused)
│ ├── features/ # (scaffold, unused)
│ ├── models/
│ │ ├── poisson_model.py # static Poisson baseline
│ │ ├── evaluate_poisson.py
│ │ ├── time_decay_poisson.py # time-decay + Dixon-Coles
│ │ ├── market_predictions.py # all 9 markets from Poisson
│ │ ├── simulate_live.py # holdout-of-N live simulation
│ │ ├── ensemble.py # equal-weight ensemble
│ │ ├── ensemble_weighted.py # validation-tuned weights
│ │ ├── ensemble_with_market.py # 4-way blend (includes market)
│ │ └── value_dc_x2.py # value detection on DC X2
│ ├── evaluation/ # (scaffold, unused)
│ └── utils/ # (scaffold, unused)
├── models/ # (early scaffold, superseded by src/models)
├── services/
│ └── football_data.py # TheSportsDB API test scripts
├── notebooks/
├── frontend/ # (to be built)
└── README.md # this file


---

## Data pipeline

### Source

Eight seasons of Premier League data from `football-data.co.uk`, covering
2018/19 through 2025/26. Each season is a single CSV with results, match
statistics (shots, shots on target, corners, cards), and bookmaker odds.

### Canonical master file

`data/premier_league_master.csv` is built by `combine_data.py` from the eight
raw season files.

**Two facts every downstream script must get right** — both were the source
of real bugs:

- `Date` is stored as an **ISO string, `%Y-%m-%d`** (explicitly parsed from
  the raw `%d/%m/%Y` format during the master rebuild, then re-saved as ISO).
  Parsing it with any other format string silently produces `NaT`.
- `Season` is a **4-digit code taken from the filename**, e.g. `"2526"`,
  `"2425"` — **not** `"2025/26"`.

### Final master file

- **3,040 matches**, 8 seasons
- **0 duplicates**
- **10 Aug 2018 – 24 May 2026**
- **100% coverage** on core result/statistics fields
- **~87.5% coverage** on bookmaker odds (`AvgH`, `AvgD`, `AvgA`)

### Key columns

| Column | Meaning |
|---|---|
| `Date` | ISO `%Y-%m-%d` |
| `HomeTeam`, `AwayTeam` | Team names (football-data.co.uk convention) |
| `FTHG`, `FTAG`, `FTR` | Full-time home/away goals and result (H/D/A) |
| `HS`, `AS` | Home/away shots |
| `HST`, `AST` | Home/away shots on target |
| `HC`, `AC` | Home/away corners |
| `AvgH`, `AvgD`, `AvgA` | Average bookmaker decimal odds (missing pre-2019/20) |
| `Season` | 4-digit code e.g. `"2526"` |

---

## Modelling protocol

This split is used **consistently** across every model:

| Split | Date range | Purpose |
|---|---|---|
| **Development** | `Date < 2023-08-01` | Fit model parameters |
| **Validation** | `2023-08-01 <= Date < 2025-08-01` | Tune hyperparameters (decay, K-factor, blend weights) |
| **Final test** | `Date >= 2025-08-01` (2025/26 season, 380 matches) | Evaluated **exactly once**, never used for tuning |

**Hard rule:** never tune anything against the final test season. Enforced
throughout the codebase.

**Metrics:** accuracy for classification, log loss for probability quality,
Brier score for calibration. Market benchmark uses margin-normalized
`1/AvgH`, `1/AvgD`, `1/AvgA`.

---

## Models

### 1. Elo rating

Dynamic team strength. Starts every team at 1500. Updates after each match
with `K=40`, home advantage `+40` (both tuned via grid search on validation).
Converts Elo difference to 1X2 probabilities via empirical band-counting on
historical matches with similar rating gaps (band width 75, min 50 matches).

**Walk-forward home-win accuracy** (binary H vs not-H):

| Test season | Accuracy |
|---|---:|
| 2022/23 | 65.79% |
| 2023/24 | 66.84% |
| 2024/25 | 62.89% |
| **2025/26 (final)** | **61.32%** |

### 2. Time-decay Dixon-Coles Poisson

Per-team attack and defence parameters, fitted via maximum likelihood with:
- **Exponential time decay** (`ξ = 0.003`, tuned on validation). A match
  from 2 years ago gets ~0.11× the weight of a match from yesterday.
- **Dixon-Coles low-score correction** (`ρ` fitted per model — near zero
  in practice for this dataset).
- **Regularisation** toward attack=1, defence=1.

Produces a full score matrix (0–0 through 9–9), from which **9 markets** are
derived. Home advantage: ~1.12–1.19 depending on training window.

### 3. Logistic regression on engineered features

21 features: rolling 5-match form points, home/away-specific form, rolling
goals scored/conceded, shots, shots on target, corners, plus all corresponding
home-minus-away differences. Trained on `premier_league_features.csv`
(built with strict point-in-time discipline — no future info).

### 4. Market probabilities (benchmark, not a model)

`1 / AvgH`, `1 / AvgD`, `1 / AvgA`, normalised to remove bookmaker margin.
Serves as the benchmark that every model is compared against.

### 5. Ensembles

- **Equal-weight blend** of Elo + Poisson + ML on test: log loss 1.0294.
- **Validation-tuned weights**: 0.75 Elo / 0.20 Poisson / 0.05 ML.
  Validation log loss 0.9654, test log loss 1.0260.
- **Four-way blend including market**: grid search assigns **100% weight to
  the market**. The market alone is a better probability estimator than any
  non-zero blend of the other three.

---

## Results — 2025/26 final test (380 matches)

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

Evaluated on 380 test matches. Base rate = "always predict the majority
class for that market."

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

### Confidence-tier accuracy

Markets with high base rates produce high-confidence predictions that hit
70–95%:

| Market | Top 10% confidence (38 picks) | Top 25% (95 picks) |
|---|---:|---:|
| Home Team to Score | 92.1% | 94.7% |
| Double Chance 1X | 89.5% | 87.4% |
| Away Team to Score | 89.5% | 81.1% |
| Double Chance X2 | 84.2% | 75.8% |
| Over 1.5 | 81.6% | 84.2% |

**These numbers are real but not edge.** They reflect that when the model
is confident, the outcome is often genuinely high-probability. The market
identifies the same matches.

### Value detection — the decisive test

For Double Chance X2 (the one market where the model appeared to beat the
base rate), we tested whether it beat the **market**:

| Threshold (model − market) | N picks | Model avg | Market avg | Actual | ROI |
|---:|---:|---:|---:|---:|---:|
| ≥ 2% | 187 | 0.574 | 0.489 | 0.471 | **−7.80%** |
| ≥ 5% | 132 | 0.579 | 0.475 | 0.485 | **−4.73%** |
| ≥ 10% | 52 | 0.594 | 0.438 | 0.423 | **−2.50%** |
| ≥ 15% | 24 | 0.621 | 0.422 | 0.292 | **−35.19%** |

**At every threshold, the market was closer to the actual outcome than the
model.** On the 24 matches where the model claimed the largest edge, it
predicted 62.1% and actual was 29.2% — off by 33 percentage points. The
market was off by 13.

**Conclusion: the model has no exploitable edge over the closing line.**

---

## What we tried — and what worked or didn't

| Upgrade | Verdict |
|---|---|
| **Time decay** on Poisson | ✅ Real improvement (log loss 1.0494 → 1.0404) |
| **Dixon-Coles correction** | ➖ Flat (1.0405 → 1.0404). Not worth it. |
| **Home/away split** (4 params per team) | ❌ Regression on validation (0.9897 → 1.0069) |
| **Equal-weight ensemble** | ✅ Small gain (1.0316 → 1.0294) |
| **Weighted ensemble** | ✅ Small additional gain (1.0294 → 1.0260) |
| **Adding market to blend** | ❌ Market takes 100% weight |
| **Value betting on DC X2** | ❌ Negative ROI at every threshold |
| **Elo 1X2 via polynomial mapping** | ❌ Overfit; worse than empirical band-counting |
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
   predictions hit 84–94% on several markets, but that's because the market
   also assigns high confidence to those same matches.

4. **Calibration is more important than accuracy.** Log loss exposed the
   failures (zero-draw prediction, home-bias) that accuracy alone hid.

5. **The market is a hard benchmark, and that's fine.** A model that
   matches the market on football-only data is a legitimate result. Beating
   it requires information the model doesn't have.

---

## Known pitfalls

- **Do not paste large scripts into Terminal via `cat > file <<'PY' ... PY`.**
  Terminal.app chokes on large pastes (shell completions, redraw glitches).
  Use `open -e` with TextEdit for files over ~100 lines.
- **Always state the `Date` format explicitly** when parsing — never let
  pandas guess. Silent `NaT` corruption is the classic failure mode.
- **Season filters use `"2526"`-style codes**, not `"2025/26"`.
- **Never tune a hyperparameter against the final test season.**
- **`AS` is a SQL keyword.** Quote it as `"AS"` in DuckDB queries.

---

## Suggested next steps

1. **Build the web interface.** FastAPI backend that takes a fixture and
   returns the full market card. Simple frontend that displays probabilities
   per match. Uses existing model outputs — no new modelling required.
2. **Initialize git** (`git init` + commit) so future sessions have real
   checkpoints instead of relying on a chat transcript.
3. **Optional: reframe as a value-detection/analytics tool** rather than a
   betting engine. The honest pitch is "given a fixture, show me the model's
   probability and compare it to the market," not "find winning bets."

---

## Project status

**Modelling: complete and honest.** The pipeline is clean, walk-forward
validated, benchmarked, and the results are what they are.

**Next: web interface** to make the model usable and visible.

---

*Last updated: after value-detection test on DC X2 confirmed no exploitable
edge over closing odds.*