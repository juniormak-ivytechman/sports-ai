"use client";

import { useState } from "react";
import { MarketCard as MarketCardType, fetchExplanation } from "@/lib/api";

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

function ProbabilityBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium">{label}</span>
        <span className="font-mono">{pct(value)}</span>
      </div>
      <div className="w-full bg-gray-200 rounded h-2">
        <div className={`${color} h-2 rounded`} style={{ width: pct(value) }} />
      </div>
    </div>
  );
}

export default function MarketCard({ card }: { card: MarketCardType }) {
  const m = card.markets;
  const [explanation, setExplanation] = useState<string | null>(null);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);

  async function handleExplain() {
    setLoadingExplain(true);
    setExplainError(null);
    try {
      const res = await fetchExplanation(card.homeTeam, card.awayTeam);
      setExplanation(res.explanation);
    } catch (e: any) {
      setExplainError(e.message || "Failed to load explanation.");
    } finally {
      setLoadingExplain(false);
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-4">
      <div className="flex justify-between items-baseline mb-4">
        <h2 className="text-xl font-bold">
          {card.homeTeam} vs {card.awayTeam}
        </h2>
        <span className="text-sm text-gray-500">
          xG: {card.expectedHomeGoals.toFixed(2)} – {card.expectedAwayGoals.toFixed(2)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-semibold text-gray-600 mb-2">1X2</h3>
          <ProbabilityBar label="Home" value={m["1X2"].home} color="bg-blue-500" />
          <ProbabilityBar label="Draw" value={m["1X2"].draw} color="bg-gray-500" />
          <ProbabilityBar label="Away" value={m["1X2"].away} color="bg-red-500" />

          <h3 className="text-sm font-semibold text-gray-600 mt-4 mb-2">Double Chance</h3>
          <ProbabilityBar label="1X" value={m.dc_1x} color="bg-blue-400" />
          <ProbabilityBar label="X2" value={m.dc_x2} color="bg-red-400" />
          <ProbabilityBar label="12" value={m.dc_12} color="bg-purple-400" />
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-600 mb-2">Goals</h3>
          <ProbabilityBar label="Over 1.5" value={m.over_1_5} color="bg-green-500" />
          <ProbabilityBar label="Over 2.5" value={m.over_2_5} color="bg-green-600" />
          <ProbabilityBar label="Over 3.5" value={m.over_3_5} color="bg-green-700" />

          <h3 className="text-sm font-semibold text-gray-600 mt-4 mb-2">BTTS / Teams</h3>
          <ProbabilityBar label="BTTS Yes" value={m.btts_yes} color="bg-yellow-500" />
          <ProbabilityBar label="Home to score" value={m.home_to_score} color="bg-blue-300" />
          <ProbabilityBar label="Away to score" value={m.away_to_score} color="bg-red-300" />
        </div>
      </div>

      <div className="mt-6 pt-4 border-t">
        {!explanation && !loadingExplain && (
          <button
            onClick={handleExplain}
            className="bg-gray-900 text-white text-sm px-4 py-2 rounded hover:bg-gray-700 transition"
          >
            ✨ Explain this prediction
          </button>
        )}
        {loadingExplain && (
          <p className="text-sm text-gray-500 italic">
            Generating explanation… (first call may take ~30s if backend was asleep)
          </p>
        )}
        {explainError && (
          <p className="text-sm text-red-600">Error: {explainError}</p>
        )}
        {explanation && (
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Analyst&apos;s take
            </h3>
            <p className="text-gray-800 leading-relaxed">{explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}