Financial Analyst MVP – SEC Filings RAG + CrewAI

-An end-to-end AI-powered financial analyst system that:

-Scrapes real SEC EDGAR filings (8-K, 10-Q)

-Cleans and chunks filings into structured text

-Stores embeddings in a FAISS vector database

-Uses Retrieval-Augmented Generation (RAG) to answer analyst-style questions

-Generates a daily financial brief

-Orchestrates the entire workflow using CrewAI agents

This project demonstrates practical LLM system design, data ingestion, vector search, and agent-based automation.
Features

-Live SEC EDGAR scraping (AAPL, MSFT, NVDA, TSLA, AMZN by default)

-Incremental FAISS indexing (only embeds new documents)

-RAG Q&A grounded in real filings (no hallucinated summaries)

-Daily analyst-style report with source links

-CrewAI multi-agent orchestration (Scraper → Indexer → Analyst)

-Optional scheduled automation (cron-ready, disabled by default)

System Architecture

SEC EDGAR
↓
Scrapers
↓
Cleaned Docs
↓
Chunking
↓
Embeddings (OpenAI)
↓
FAISS Vector Store
↓
RAG Pipeline
↓
Daily Financial Brief
↓
CrewAI Agent Orchestration

Project Structure

financial-analyst-mvp/
├── agents/ # CrewAI agents and workflow
├── automation/ # Daily report + Crew runner
├── embeddings/ # Chunking, FAISS build/update
├── preprocess/ # Document cleaning
├── rag_pipeline/ # RAG query interface
├── scripts/ # SEC scrapers
├── reports/ # Generated reports (gitignored)
├── .env.example # Environment variable template
├── requirements.txt
└── README.md

Setup
1. Clone the repository
git clone https://github.com/amoghmakam/financial-analyst-mvp.git
cd financial-analyst-mvp

2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Add your OpenAI API key
cp .env.example .env

Edit .env and add your key:
OPENAI_API_KEY=your_real_api_key_here
TICKERS=AAPL,MSFT,NVDA,TSLA,AMZN

Usage

Generate a daily report (without agents):
python automation/daily_report.py

output:
reports/daily_YYYY-MM-DD.txt

Run the full CrewAI workflow:
python -m automation.run_with_crewai


Ask a direct RAG question:
python rag_pipeline/ask.py "What did Amazon disclose in its most recent 8-K?" --ticker AMZN --doc-type 8-K --most-recent


Scheduling (Optional)

The system is cron-ready for daily execution (disabled by default): 30 8 * * * python -m automation.run_with_crewai


Future Improvements

Financial news ingestion (Yahoo Finance, CNBC)

Sentiment analysis per company

Stock price correlation with filings

Slack or email report delivery

Web UI (Streamlit or Gradio)
