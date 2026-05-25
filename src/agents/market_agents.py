from src.ingestion.news_agent import fetch_news
from src.ingestion.rss_agent import fetch_rss_news
from src.rag.vector_store import index_articles
from src.rag.research_copilot import answer_question

# Fallback RSS feeds used when NEWS_API_KEY is absent (no key required)
_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]


def _normalize_news_row(row: dict) -> dict:
    """Map fetch_news() column names to the schema index_articles() expects."""
    return {
        "title":        row.get("headline", ""),
        "description":  row.get("summary", ""),
        "content":      row.get("content", ""),
        "source":       row.get("source", ""),
        "url":          row.get("url", ""),
        "published_at": row.get("published_at", ""),
    }


def _normalize_rss_row(row: dict) -> dict:
    """Map fetch_rss_news() column names to the schema index_articles() expects."""
    return {
        "title":        row.get("headline", ""),
        "description":  row.get("summary", ""),
        "content":      "",
        "source":       row.get("source", "RSS"),
        "url":          row.get("url", ""),
        "published_at": row.get("published_at", ""),
    }


def run_market_intelligence_workflow(topic: str = "AI Infrastructure") -> dict:
    """
    Agentic workflow:
    1. Fetch live news via NewsAPI (primary) or RSS feeds (fallback, no key needed)
    2. Normalize articles to a consistent schema
    3. Index into ChromaDB under the correct sector metadata
    4. Query the RAG copilot for a source-grounded market summary
    5. Return summary, evidence, and source list
    """
    articles: list[dict] = []

    # Primary: NewsAPI
    news_df = fetch_news(topic, page_size=10)
    if news_df is not None and not news_df.empty:
        articles = [_normalize_news_row(r) for r in news_df.to_dict(orient="records")]

    # Fallback: RSS (free, no key required)
    if not articles:
        for feed_url in _RSS_FEEDS:
            try:
                rss_df = fetch_rss_news(feed_url, limit=5)
                if not rss_df.empty:
                    articles.extend(
                        _normalize_rss_row(r) for r in rss_df.to_dict(orient="records")
                    )
            except Exception:
                continue

    # Index into ChromaDB with correct sector metadata
    indexed_count = index_articles(articles, sector=topic) if articles else 0

    fallback_summary = (
        f"No live news was retrieved for '{topic}'. "
        "Verify NEWS_API_KEY is configured, or check your internet connection."
    )

    question = (
        f"Summarize the top market signals, opportunities, risks, "
        f"and investor watchouts for {topic}."
    )

    summary, evidence = answer_question(
        question=question,
        sector=topic,
        fallback_summary=fallback_summary,
    )

    sources = [
        {
            "title":  a.get("title", "Untitled"),
            "source": a.get("source", "Unknown source"),
            "url":    a.get("url", ""),
        }
        for a in articles
    ]

    return {
        "topic":           topic,
        "articles_loaded": len(articles),
        "records_indexed": indexed_count,
        "summary":         summary,
        "evidence":        evidence,
        "sources":         sources,
    }
