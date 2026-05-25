# MarketPulse AI

GenAI-powered Alternative Data Intelligence Platform prototype for capital markets use cases.

## What this prototype includes

- Streamlit executive dashboard
- Alternative data explorer
- Sample market intelligence dataset
- Optional live news ingestion through NewsAPI
- Market data agent using Alpha Vantage API
- Baseline sentiment engine
- ChromaDB vector store
- RAG research copilot
- Executive summary agent
- Compliance/governance disclaimer agent
- VS Code launch configuration

## Quick start

```bash
cd marketpulse-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional: enable GenAI and live data

```bash
cp .env.example .env
```

Then add:

```bash
OPENAI_API_KEY=your_key_here
NEWS_API_KEY=your_newsapi_key_here
```

The app runs without keys using sample data and a transparent keyword-based sentiment model.

## Recommended demo flow

1. Open Executive Dashboard
2. Select AI Infrastructure
3. Show sentiment, themes, and risk alerts
4. Open Alternative Data Explorer
5. Show source evidence
6. Open RAG Admin and index the sector
7. Open AI Research Copilot and ask: "What are the top risks in AI infrastructure?"
8. Open Executive Insight Generator and create a client-ready note

## Project structure

```text
marketpulse-ai/
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   └── sample_market_data.csv
├── docs/
├── src/
│   ├── agents/
│   ├── ingestion/
│   ├── rag/
│   ├── sentiment/
│   └── utils/
└── .vscode/
```

## Roadmap

### MVP 1
- Run with sample data
- Show dashboard and source evidence
- Use local ChromaDB RAG

### MVP 2
- Add live news ingestion
- Add Reddit ingestion
- Add SEC filings and earnings transcripts

### MVP 3
- Add multi-agent orchestration
- Add user feedback loop
- Add model evaluation metrics
- Add explainability scoring

## Governance note

This prototype is for market intelligence exploration only and does not provide investment advice.
