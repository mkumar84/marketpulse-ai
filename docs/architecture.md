# Architecture

```text
Data Sources
  ├── Sample CSV
  ├── NewsAPI
  ├── RSS feeds
  ├── Alpha Vantage (daily OHLCV)
  └── Future: Reddit / SEC / earnings transcripts

Ingestion Agents
  ↓
Sentiment Engine
  ↓
ChromaDB Vector Store
  ↓
RAG Research Copilot
  ↓
Executive Summary Agent
  ↓
Streamlit UI
```

## Agents

- News Agent: collects live news when configured
- Market Data Agent: retrieves 5-day ticker performance
- Sentiment Agent: classifies bullish, bearish, neutral signals
- RAG Agent: retrieves source evidence
- Executive Summary Agent: creates boardroom-ready notes
- Compliance Agent: adds governance disclaimers
