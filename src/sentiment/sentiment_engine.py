POSITIVE = [
    # Growth & performance
    "growth", "increase", "strong", "bullish", "expansion", "investment", "adoption",
    "rise", "opportunity", "record", "beat", "exceed", "outperform", "upgrade",
    "momentum", "resilient", "recovery", "rebound", "milestone", "accelerate",
    "partnership", "acquisition", "gain", "profit", "revenue", "demand",
    # Capital markets specific
    "inflows", "rally", "surge", "robust", "positive", "improve", "advance",
    "breakthrough", "launch", "leadership", "efficiency", "savings", "returns",
    "confidence", "stable", "diversify", "innovative", "pioneer",
]

NEGATIVE = [
    # Risk & decline — note: "risk" excluded as it is domain-neutral in finance
    "shortage", "pressure", "weak", "bearish", "decline", "constraint", "slow",
    "bottleneck", "concern", "miss", "downgrade", "underperform", "loss",
    "uncertainty", "headwind", "write-down", "impairment", "delinquency", "default",
    "investigation", "lawsuit", "fraud", "cut", "layoff", "restructure",
    # Capital markets specific
    "selloff", "correction", "slump", "deteriorate", "tighten", "negative",
    "fall", "drop", "crash", "fail", "warning", "recall", "caution", "cautious",
    "stress", "exposed", "vulnerable", "worsen", "squeeze",
]


def score_text(text: str) -> tuple[str, float]:
    """Keyword-based sentiment baseline. Domain-neutral terms like 'risk' are excluded to avoid bias."""
    text_lower = (text or "").lower()
    pos = sum(1 for word in POSITIVE if word in text_lower)
    neg = sum(1 for word in NEGATIVE if word in text_lower)
    raw = pos - neg
    score = max(min(raw / 5, 1.0), -1.0)
    if score > 0.15:
        return "Bullish", round(score, 2)
    if score < -0.15:
        return "Bearish", round(score, 2)
    return "Neutral", round(score, 2)


def sentiment_label(avg_score: float) -> str:
    if avg_score > 0.3:
        return "Bullish"
    if avg_score < -0.2:
        return "Bearish"
    return "Neutral"
