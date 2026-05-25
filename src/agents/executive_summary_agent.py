from openai import OpenAI
from src.utils.config import OPENAI_API_KEY
from src.sentiment.sentiment_engine import sentiment_label


def generate_summary(sector_df, sector: str) -> str:
    top_themes = ", ".join(sector_df["theme"].dropna().unique())
    top_risks = ", ".join(sector_df["risk"].dropna().unique())
    avg_score = sector_df["score"].mean()
    sentiment = sentiment_label(avg_score)

    baseline = f"""
**MarketPulse AI Summary — {sector}**

Overall sentiment is **{sentiment}** with an average signal score of **{avg_score:.2f}**.

Key emerging themes include **{top_themes}**.

Main risks to monitor include **{top_risks}**.

Executive insight: Alternative data signals suggest investors should monitor demand growth, infrastructure constraints, sentiment shifts, and near-term risk factors.
"""

    if not OPENAI_API_KEY:
        return baseline + "\n\n_Add OPENAI_API_KEY in .env to enable full GenAI narrative generation._"

    client = OpenAI(api_key=OPENAI_API_KEY)
    evidence = "\n".join(
        f"- {r['headline']} | Sentiment: {r['sentiment']} | Theme: {r['theme']} | Risk: {r['risk']}"
        for _, r in sector_df.iterrows()
    )
    prompt = f"""
Create a concise executive market note for a capital markets audience.
Sector: {sector}
Evidence:
{evidence}

Structure:
1. Executive summary
2. Top signals
3. Risks to monitor
4. Suggested client conversation angle
5. Confidence score
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
