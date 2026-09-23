"use client";

import { useEffect, useState } from "react";
import { fetchTodaysFixtures, FixturesResponse, MarketCard as MarketCardType } from "@/lib/api";
import MarketCard from "@/components/MarketCard";

export default function Home() {
  const [data, setData] = useState<FixturesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTodaysFixtures()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">Sports AI</h1>
        <p className="text-gray-600 mb-8">
          Premier League predictions — {data?.date ?? "loading"}
        </p>

        {loading && <p>Loading fixtures...</p>}
        {error && <p className="text-red-600">Error: {error}</p>}

        {data?.note && (
          <p className="text-yellow-700 bg-yellow-50 p-3 rounded mb-4">
            {data.note}
          </p>
        )}

        {data?.matches.map((fixture) => {
          const pred = fixture.prediction;
          if ("error" in pred) {
            return (
              <div key={fixture.id} className="bg-white rounded shadow p-4 mb-4">
                <p className="text-gray-700">
                  {fixture.id} — prediction unavailable
                </p>
              </div>
            );
          }
          return <MarketCard key={fixture.id} card={pred as MarketCardType} />;
        })}

        {data && data.matches.length === 0 && (
          <p className="text-gray-500">
            No EPL fixtures today. Check back on a match day.
          </p>
        )}
      </div>
    </main>
  );
}