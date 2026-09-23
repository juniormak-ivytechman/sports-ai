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

  const isFallback = data?.note?.includes("most recent") ?? false;

  return (
    <main className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2 text-foreground">Sports AI</h1>
        <p className="text-muted mb-8">
          Premier League predictions
          {data ? ` — ${data.date}` : " — loading"}
        </p>

        {loading && <p className="text-muted">Loading fixtures...</p>}
        {error && <p className="text-red-500">Error: {error}</p>}

        {data?.note && (
          <div className="bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-200 p-3 rounded mb-4 text-sm">
            {data.note}
          </div>
        )}

        {isFallback && (
          <p className="text-muted text-sm mb-4">
            No live fixtures right now — showing the most recent matchday instead.
          </p>
        )}

        {data?.matches.map((fixture) => {
          const pred = fixture.prediction;
          if ("error" in pred) {
            return (
              <div key={fixture.id} className="bg-card-bg rounded shadow p-4 mb-4 border border-card-border">
                <p className="text-muted">Prediction unavailable for match {fixture.id}</p>
              </div>
            );
          }
          return <MarketCard key={fixture.id} card={pred as MarketCardType} />;
        })}

        {data && data.matches.length === 0 && (
          <p className="text-muted">
            No EPL fixtures today. Check back on a match day.
          </p>
        )}
      </div>
    </main>
  );
}