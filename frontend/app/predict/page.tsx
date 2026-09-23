"use client";

import { useEffect, useState } from "react";
import { fetchTeams, fetchPrediction, MarketCard as MarketCardType } from "@/lib/api";
import MarketCard from "@/components/MarketCard";

export default function PredictPage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [card, setCard] = useState<MarketCardType | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchTeams().then(setTeams).catch(console.error);
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!home || !away) return;
    setLoading(true);
    try {
      const result = await fetchPrediction(home, away);
      setCard(result);
    } catch (err) {
      alert("Could not predict this fixture. Teams might not be in the model.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Predict a Fixture</h1>

        <form onSubmit={handleSubmit} className="bg-white rounded shadow p-6 mb-8">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <select
              value={home}
              onChange={(e) => setHome(e.target.value)}
              className="border rounded p-2"
            >
              <option value="">Home team...</option>
              {teams.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <select
              value={away}
              onChange={(e) => setAway(e.target.value)}
              className="border rounded p-2"
            >
              <option value="">Away team...</option>
              {teams.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <button
            type="submit"
            disabled={loading || !home || !away}
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {card && <MarketCard card={card} />}
      </div>
    </main>
  );
}