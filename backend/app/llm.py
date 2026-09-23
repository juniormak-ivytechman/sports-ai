import os
from groq import Groq

_client = None

def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            return None
        _client = Groq(api_key=key)
    return _client


def explain_prediction(prediction: dict, home_form: dict = None, away_form: dict = None,
                       home_injuries: list = None, away_injuries: list = None) -> str:
    """Generate a plain-English explanation of a model prediction.

    The LLM receives STRUCTURED data from our model and writes prose.
    It must NOT invent statistics — only interpret the ones provided.
    """
    client = _get_client()
    if client is None:
        return "LLM explanation unavailable: GROQ_API_KEY not configured."

    m = prediction["markets"]
    one_x_two = m["1X2"]

    # Build a compact fact sheet for the LLM
    facts = [
        f"Match: {prediction['homeTeam']} (home) vs {prediction['awayTeam']} (away)",
        f"Expected goals: {prediction['homeTeam']} {prediction['expectedHomeGoals']:.2f}, "
        f"{prediction['awayTeam']} {prediction['expectedAwayGoals']:.2f}",
        f"Win probabilities: {prediction['homeTeam']} {one_x_two['home']*100:.1f}%, "
        f"Draw {one_x_two['draw']*100:.1f}%, {prediction['awayTeam']} {one_x_two['away']*100:.1f}%",
        f"Over 1.5 goals: {m['over_1_5']*100:.1f}%",
        f"Over 2.5 goals: {m['over_2_5']*100:.1f}%",
        f"Both teams to score: {m['btts_yes']*100:.1f}%",
        f"{prediction['homeTeam']} to score: {m['home_to_score']*100:.1f}%",
        f"{prediction['awayTeam']} to score: {m['away_to_score']*100:.1f}%",
        f"Double chance 1X ({prediction['homeTeam']} or draw): {m['dc_1x']*100:.1f}%",
        f"Double chance X2 (draw or {prediction['awayTeam']}): {m['dc_x2']*100:.1f}%",
    ]

    if home_form:
        facts.append(
            f"{prediction['homeTeam']} recent form: {home_form.get('summary', 'n/a')}"
        )
    if away_form:
        facts.append(
            f"{prediction['awayTeam']} recent form: {away_form.get('summary', 'n/a')}"
        )
    if home_injuries:
        facts.append(
            f"{prediction['homeTeam']} injury/suspension news: "
            + "; ".join(home_injuries[:5])
        )
    if away_injuries:
        facts.append(
            f"{prediction['awayTeam']} injury/suspension news: "
            + "; ".join(away_injuries[:5])
        )

    facts_str = "\n".join(f"- {f}" for f in facts)

    system_prompt = (
        "You are a football analytics explainer. You receive structured model "
        "output and produce a concise, plain-English explanation for a football "
        "fan. Rules you MUST follow:\n"
        "1. Only use the numbers and facts provided. Never invent statistics, "
        "player names, scores, or events.\n"
        "2. Write 3-4 sentences. No bullet points. No headers.\n"
        "3. Explain WHICH outcomes the model favours and WHY, based on the "
        "probabilities and expected goals.\n"
        "4. If injury news is provided, mention it briefly. If not, don't "
        "speculate about injuries.\n"
        "5. Sound like a knowledgeable analyst, not a robot. Use natural "
        "language.\n"
        "6. Do NOT recommend bets. This is an analysis tool, not betting advice."
    )

    user_prompt = (
        f"Here is the model output for a Premier League match:\n\n"
        f"{facts_str}\n\n"
        f"Write the explanation now."
    )


    MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ]

    last_error = None
    for model_id in MODELS:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=1500,
            )
            msg = response.choices[0].message

            # Reasoning models may put output in `content` or `reasoning`
            content = (msg.content or "").strip()
            if not content:
                # Try the reasoning field as fallback
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    content = reasoning.strip()

            print(f"[llm] model={model_id} "
                  f"finish={response.choices[0].finish_reason} "
                  f"content_len={len(msg.content or '')} "
                  f"reasoning_len={len(getattr(msg, 'reasoning', '') or '')}")

            if content:
                return content
            # If content is empty, try next model
            continue

        except Exception as e:
            last_error = e
            print(f"[llm] model={model_id} failed: {e}")
            continue

    return f"LLM error: {str(last_error)}"