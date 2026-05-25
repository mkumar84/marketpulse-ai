def detect_top_themes(df, top_n: int = 5) -> list[str]:
    """Return the top N themes by signal frequency."""
    return df["theme"].dropna().value_counts().head(top_n).index.tolist()


def detect_risks(df, top_n: int = 5) -> list[str]:
    """Return the top N risks by signal frequency."""
    return df["risk"].dropna().value_counts().head(top_n).index.tolist()
