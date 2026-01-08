import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- Basic setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

st.set_page_config(page_title="Financial Analyst MVP", layout="wide")

st.title("📈 Financial Analyst MVP — SEC Filings RAG (FAISS) + CrewAI")
st.caption("Localhost UI for asking questions over SEC filings using your existing RAG pipeline.")


def run_cmd(args_list):
    """Run a command in the project root, return stdout (or raise with stderr)."""
    proc = subprocess.run(
        args_list,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "Command failed")
    return proc.stdout


def parse_answer_and_sources(output: str):
    """Extract === Answer === and === Sources === blocks from ask.py output."""
    answer = ""
    sources = []

    m_ans = re.search(r"=== Answer ===\s*\n\n(.*?)(?:\n\n=== Sources|\Z)", output, re.S)
    if m_ans:
        answer = m_ans.group(1).strip()

    m_src = re.search(r"=== Sources \(top hits\) ===\s*\n(.*)\Z", output, re.S)
    if m_src:
        lines = [ln.strip() for ln in m_src.group(1).splitlines() if ln.strip()]
        # lines look like: "- https://..."
        for ln in lines:
            ln = ln.lstrip("-").strip()
            if ln.startswith("http"):
                sources.append(ln)

    return answer, sources


# --- Sidebar controls ---
st.sidebar.header("Controls")

default_tickers = os.getenv("TICKERS", "AAPL,MSFT,NVDA,TSLA,AMZN")
ticker_list = [t.strip() for t in default_tickers.split(",") if t.strip()]
ticker_options = ["ALL"] + ticker_list

ticker = st.sidebar.selectbox("Ticker", ticker_options, index=0)
doc_type = st.sidebar.selectbox("Doc Type", ["8-K", "10-Q", "10-K"], index=0)

most_recent = st.sidebar.checkbox("Most recent filing only", value=True)
k = st.sidebar.slider("Top-K retrieval (k)", min_value=10, max_value=200, value=80, step=10)
max_chunks = st.sidebar.slider("Max chunks to summarize", min_value=3, max_value=20, value=10, step=1)

model = st.sidebar.text_input("Model (optional)", value="gpt-4o-mini")
embed_model = st.sidebar.text_input("Embedding model (optional)", value="text-embedding-3-small")

st.sidebar.divider()
st.sidebar.subheader("Automation")
run_daily = st.sidebar.button("📰 Generate Daily Report")
run_crew = st.sidebar.button("🤖 Run CrewAI Workflow")


# --- Main: Ask a question ---
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Ask a question")
    question = st.text_area(
        "Query",
        value="What did Amazon disclose in its most recent 8-K?",
        height=90,
    )

    ask = st.button("🔎 Ask", type="primary")

with col2:
    st.subheader("What this UI does")
    st.markdown(
        """
- Uses your existing `rag_pipeline/ask.py`
- Shows answer + source links
- Runs locally on your machine
        """
    )
    st.info("Tip: Use **Most recent** + a specific ticker for the best answers.")


# --- Handle buttons ---
def ensure_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to your .env file before running.")
        st.stop()


if ask:
    ensure_api_key()
    cmd = [sys.executable, "rag_pipeline/ask.py", question, "--k", str(k), "--max-chunks", str(max_chunks)]

    if ticker != "ALL":
        cmd += ["--ticker", ticker]

    if doc_type:
        cmd += ["--doc-type", doc_type]

    if most_recent:
        cmd += ["--most-recent"]

    # optional args if your script supports them (safe to include if you implemented these flags)
    # If your ask.py doesn't accept them, remove these two blocks.
    if model:
        cmd += ["--chat-model", model]
    if embed_model:
        cmd += ["--embed-model", embed_model]

    with st.spinner("Running RAG query..."):
        try:
            out = run_cmd(cmd)
            answer, sources = parse_answer_and_sources(out)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader("✅ Answer")
    st.write(answer or "No answer returned.")

    st.subheader("🔗 Sources")
    if sources:
        for s in sources:
            st.markdown(f"- {s}")
    else:
        st.write("No sources parsed. (The answer may still be valid — check the raw output below.)")

    with st.expander("Raw output (debug)"):
        st.code(out)


if run_daily:
    ensure_api_key()
    with st.spinner("Generating daily report..."):
        try:
            out = run_cmd([sys.executable, "automation/daily_report.py"])
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("Daily report generated.")
    st.code(out.strip() or "(ok)")

    # Show the newest report if it exists
    reports_dir = PROJECT_ROOT / "reports"
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("daily_*.txt"))
        if reports:
            latest = reports[-1]
            st.subheader("Latest report")
            st.caption(str(latest))
            st.text(latest.read_text(encoding="utf-8")[:4000])


if run_crew:
    ensure_api_key()
    with st.spinner("Running CrewAI workflow (scrape → clean → index → report)..."):
        try:
            out = run_cmd([sys.executable, "-m", "automation.run_with_crewai"])
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("CrewAI workflow finished.")
    st.code(out.strip() or "(ok)")

