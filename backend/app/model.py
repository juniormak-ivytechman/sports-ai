import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_FILE = str(_REPO_ROOT / "data" / "premier_league_master.csv")

DECAY = 0.003
REGULARIZATION = 0.01
MAX_ITER = 500
SCORE_MATRIX_SIZE = 10
RHO_BOUNDS = (-0.20, 0.20)


class PoissonModel:
    def __init__(self, master_path=MASTER_FILE):
        self.df = pd.read_csv(master_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%Y-%m-%d")
        self.df = self.df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

        self.atk = None
        self.dfn = None
        self.ha = None
        self.rho = None
        self.t2i = {}

    def fit(self):
        """Fit time-decay Dixon-Coles Poisson on all historical data."""
        training = self.df
        teams = sorted(set(training["HomeTeam"]) | set(training["AwayTeam"]))
        t2i = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = training["HomeTeam"].map(t2i).to_numpy(int)
        ai = training["AwayTeam"].map(t2i).to_numpy(int)
        hg = training["FTHG"].to_numpy(float)
        ag = training["FTAG"].to_numpy(float)

        dates = training["Date"].to_numpy("datetime64[D]")
        latest = dates.max()
        age = (latest - dates).astype("timedelta64[D]").astype(float)

        cf = gammaln(hg + 1) + gammaln(ag + 1)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def unpack(p):
            return np.exp(p[:n]), np.exp(p[n:2*n]), np.exp(p[-2]), p[-1]

        def obj(p, w):
            atk, dfn, ha, rho = unpack(p)
            hl = np.clip(ha * atk[hi] * dfn[ai], 1e-8, None)
            al = np.clip(atk[ai] * dfn[hi], 1e-8, None)
            base = hg * np.log(hl) - hl + ag * np.log(al) - al - cf
            tau = np.ones_like(hg)
            tau[m00] = 1 - hl[m00] * al[m00] * rho
            tau[m01] = 1 + hl[m01] * rho
            tau[m10] = 1 + al[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-10, None)
            ll = base + np.log(tau)
            loss = -np.sum(w * ll)
            reg = REGULARIZATION * (
                np.sum(p[:n]**2) + np.sum(p[n:2*n]**2) + p[-2]**2 + p[-1]**2
            )
            return loss + reg

        w = np.exp(-DECAY * age)
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [np.log(1.10), 0.0]])
        bounds = [(None, None)] * (2*n) + [(None, None), RHO_BOUNDS]

        res = minimize(obj, x0, args=(w,), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": MAX_ITER, "ftol": 1e-10,
                                "gtol": 1e-6, "maxls": 40})
        if not res.success:
            raise RuntimeError(f"Poisson fit failed: {res.message}")

        self.atk, self.dfn, self.ha, self.rho = unpack(res.x)
        self.t2i = t2i
        print(f"[model] Fitted Poisson. {n} teams. HA={self.ha:.3f} rho={self.rho:+.4f}")

    def score_matrix(self, hl, al):
        g = np.arange(SCORE_MATRIX_SIZE)
        hp = np.exp(-hl + g * np.log(max(hl, 1e-8)) - gammaln(g + 1))
        ap = np.exp(-al + g * np.log(max(al, 1e-8)) - gammaln(g + 1))
        m = np.outer(hp, ap)
        m[0, 0] *= max(1 - hl * al * self.rho, 1e-10)
        m[0, 1] *= max(1 + hl * self.rho, 1e-10)
        m[1, 0] *= max(1 + al * self.rho, 1e-10)
        m[1, 1] *= max(1 - self.rho, 1e-10)
        return m / m.sum()

    def predict(self, home_team, away_team):
        """Return full market card dict, or None if teams unknown."""
        h, a = home_team, away_team

        # Try to map common football-data.org names to our internal names
        h_internal = self._map_team(h)
        a_internal = self._map_team(a)

        h_atk = self.atk[self.t2i[h_internal]] if h_internal in self.t2i else 1.0
        h_dfn = self.dfn[self.t2i[h_internal]] if h_internal in self.t2i else 1.0
        a_atk = self.atk[self.t2i[a_internal]] if a_internal in self.t2i else 1.0
        a_dfn = self.dfn[self.t2i[a_internal]] if a_internal in self.t2i else 1.0

        hl = max(self.ha * h_atk * a_dfn, 1e-8)
        al = max(a_atk * h_dfn, 1e-8)
        m = self.score_matrix(hl, al)

        goals = np.arange(SCORE_MATRIX_SIZE)
        total = goals[:, None] + goals[None, :]

        home_win = np.tril(m, -1).sum()
        draw = np.trace(m)
        away_win = np.triu(m, 1).sum()

        return {
            "homeTeam": h,
            "awayTeam": a,
            "expectedHomeGoals": float(hl),
            "expectedAwayGoals": float(al),
            "markets": {
                "1X2": {
                    "home": float(home_win),
                    "draw": float(draw),
                    "away": float(away_win),
                },
                "over_1_5": float(m[total >= 2].sum()),
                "over_2_5": float(m[total >= 3].sum()),
                "over_3_5": float(m[total >= 4].sum()),
                "btts_yes": float(m[1:, 1:].sum()),
                "dc_1x": float(home_win + draw),
                "dc_x2": float(draw + away_win),
                "dc_12": float(home_win + away_win),
                "dnb_home": float(home_win / (home_win + away_win)),
                "dnb_away": float(away_win / (home_win + away_win)),
                "home_to_score": float(1 - m[0, :].sum()),
                "away_to_score": float(1 - m[:, 0].sum()),
            },
        }

    @staticmethod
    def _map_team(name):
        """Map football-data.org names to football-data.co.uk names."""
        mapping = {
            "Manchester United": "Man United",
            "Manchester City": "Man City",
            "Nottingham Forest": "Nott'm Forest",
            "Wolverhampton Wanderers": "Wolves",
            "Tottenham Hotspur": "Tottenham",
            "Brighton and Hove Albion": "Brighton",
            "West Ham United": "West Ham",
            "Newcastle United": "Newcastle",
            "Leicester City": "Leicester",
            "Leeds United": "Leeds",
            "Sheffield United": "Sheffield United",
            "Crystal Palace": "Crystal Palace",
            "Aston Villa": "Aston Villa",
            "Bournemouth": "Bournemouth",
            "Brentford": "Brentford",
            "Everton": "Everton",
            "Fulham": "Fulham",
            "Arsenal": "Arsenal",
            "Chelsea": "Chelsea",
            "Liverpool": "Liverpool",
            "Burnley": "Burnley",
            "Sunderland": "Sunderland",
            "Ipswich Town": "Ipswich",
        }
        return mapping.get(name, name)