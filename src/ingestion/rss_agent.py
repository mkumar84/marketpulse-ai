import feedparser
import pandas as pd


def fetch_rss_news(feed_url: str, limit: int = 10) -> pd.DataFrame:
    """Free no-key RSS ingestion hook."""
    feed = feedparser.parse(feed_url)
    rows = []
    for entry in feed.entries[:limit]:
        rows.append({
            "source": "RSS",
            "headline": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published_at": entry.get("published", ""),
            "summary": entry.get("summary", ""),
        })
    return pd.DataFrame(rows)
