import pandas as pd
import requests
from src.utils.config import NEWS_API_KEY


def fetch_news(query: str, page_size: int = 10, sort_by: str = "publishedAt") -> pd.DataFrame:
    """
    Fetch news articles using NewsAPI.
    
    Args:
        query: Search query/topic
        page_size: Number of articles to fetch
        sort_by: Sort order (publishedAt, relevancy, popularity)
        
    Returns:
        DataFrame with article data and computed sentiment scores
    """
    if not NEWS_API_KEY:
        return pd.DataFrame()
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": sort_by,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            return pd.DataFrame()
        
        articles = data.get("articles", [])
        
        df_data = []
        for article in articles:
            df_data.append({
                "source": article.get("source", {}).get("name", "Unknown"),
                "headline": article.get("title", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "summary": article.get("description", ""),
                "content": article.get("content", ""),
                "image": article.get("urlToImage", ""),
            })
        
        df = pd.DataFrame(df_data)
        return df
    
    except requests.exceptions.RequestException:
        return pd.DataFrame()
