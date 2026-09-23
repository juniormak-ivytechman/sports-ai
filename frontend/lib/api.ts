import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface MarketCard {
  homeTeam: string;
  awayTeam: string;
  expectedHomeGoals: number;
  expectedAwayGoals: number;
  markets: {
    "1X2": { home: number; draw: number; away: number };
    over_1_5: number;
    over_2_5: number;
    over_3_5: number;
    btts_yes: number;
    dc_1x: number;
    dc_x2: number;
    dc_12: number;
    dnb_home: number;
    dnb_away: number;
    home_to_score: number;
    away_to_score: number;
  };
}

export interface Fixture {
  id: number;
  utcDate: string;
  status: string;
  prediction: MarketCard | { error: string };
}

export interface FixturesResponse {
  date: string;
  matches: Fixture[];
  note?: string;
}

export async function fetchTodaysFixtures(): Promise<FixturesResponse> {
  const r = await axios.get(`${API_BASE}/fixtures/today`);
  return r.data;
}

export async function fetchPrediction(home: string, away: string): Promise<MarketCard> {
  const r = await axios.get(`${API_BASE}/predict`, { params: { home, away } });
  return r.data;
}

export async function fetchTeams(): Promise<string[]> {
  const r = await axios.get(`${API_BASE}/teams`);
  return r.data.teams;
}
export interface ExplainResponse {
  prediction: MarketCard;
  explanation: string;
  homeInjuries: string[];
  awayInjuries: string[];
}

export async function fetchExplanation(home: string, away: string): Promise<ExplainResponse> {
  const r = await axios.get(`${API_BASE}/explain`, { params: { home, away } });
  return r.data;
}