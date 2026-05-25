from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = str(Path(__file__).resolve().parents[2] / ".chroma")
COLLECTION_NAME = "marketpulse_docs"


def _embedding_function():
    # Always use local sentence-transformers embeddings.
    # ChromaDB's OpenAIEmbeddingFunction uses the removed v0.x API and raises
    # APIRemovedInV1 when openai>=1.0 is installed. OpenAI is used only for
    # GPT-based generation (research_copilot.py), not for embeddings here.
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
    )


def delete_collection():
    """Drop and recreate the collection — use to clear stale embeddings."""
    client = chromadb.PersistentClient(path=DB_PATH)
    client.delete_collection(name=COLLECTION_NAME)
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
    )


def index_dataframe(df) -> int:
    """Index a structured sector DataFrame. IDs are sector-namespaced to prevent cross-sector collisions."""
    collection = get_collection()
    ids, docs, metas = [], [], []
    for idx, row in df.reset_index(drop=True).iterrows():
        sector_slug = (
            str(row.get("sector", "unknown"))
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )
        text = (
            f"{row.get('sector','')} | {row.get('source','')} | "
            f"{row.get('headline','')} | Theme: {row.get('theme','')} | "
            f"Risk: {row.get('risk','')}"
        )
        ids.append(f"sample-{sector_slug}-{idx}")
        docs.append(text)
        metas.append({
            "sector": str(row.get("sector", "")),
            "source": str(row.get("source", "")),
            "theme":  str(row.get("theme", "")),
            "risk":   str(row.get("risk", "")),
            "url":    str(row.get("url", "")),
        })
    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def index_articles(articles: list, sector: str = "Live Market") -> int:
    """Index live news articles fetched from NewsAPI or RSS."""
    collection = get_collection()
    ids, docs, metas = [], [], []
    for idx, article in enumerate(articles):
        title       = article.get("title", "")
        description = article.get("description", "")
        content     = article.get("content", "")
        text = f"{sector} | {article.get('source','')} | {title} | {description} | {content}"
        ids.append(f"live-{sector.lower().replace(' ','_')}-{idx}")
        docs.append(text)
        metas.append({
            "sector":       sector,
            "source":       str(article.get("source", "")),
            "theme":        "Live News",
            "risk":         "",
            "url":          str(article.get("url", "")),
            "title":        str(title),
            "published_at": str(article.get("published_at", "")),
        })
    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def query_docs(question: str, sector: str | None = None, n_results: int = 5) -> dict:
    """Query the vector store. Safe against empty collections and small result sets."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
    actual_n = min(n_results, total)
    where = {"sector": sector} if sector else None
    return collection.query(query_texts=[question], n_results=actual_n, where=where)
