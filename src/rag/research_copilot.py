from openai import OpenAI
from src.utils.config import OPENAI_API_KEY
from src.rag.vector_store import query_docs


def answer_question(question: str, sector: str, fallback_summary: str) -> tuple[str, list[str]]:
    results = query_docs(question, sector=sector, n_results=5)
    docs = results.get("documents", [[]])[0]

    if not docs:
        return (
            "I could not find enough indexed evidence for this question. Try running the live intelligence workflow first or selecting another sector.",
            [],
        )

    if not OPENAI_API_KEY:
        return (
            fallback_summary + "\n\nNote: Add OPENAI_API_KEY in .env to enable full GenAI answers.",
            docs,
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    context = "\n".join(f"- {doc}" for doc in docs)

    prompt = f"""
You are MarketPulse AI, a capital markets research copilot for analysts, strategists, and institutional client teams.

Your task:
Answer the user's question using ONLY the evidence provided below.

Rules:
- Do not make unsupported claims.
- If evidence is limited, clearly say so.
- Use an executive-ready tone.
- Include 3 sections:
  1. Direct Answer
  2. Key Evidence
  3. Risks / Watchouts
- End with a confidence level: High, Medium, or Low.

Sector: {sector}

Question:
{question}

Evidence:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a source-grounded capital markets AI assistant. You avoid unsupported financial advice.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content, docs