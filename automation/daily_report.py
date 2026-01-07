import re
import argparse
import os
from datetime import datetime
from pathlib import Path
import subprocess

TICKERS_DEFAULT = "AAPL,MSFT,NVDA,TSLA,AMZN"

DAILY_QUESTIONS = [
    ("Market-moving disclosures (AMZN)", 'What did Amazon disclose in its most recent 8-K?', "AMZN", "8-K"),
    ("Market-moving disclosures (AAPL)", 'What did Apple disclose in its most recent 8-K?', "AAPL", "8-K"),
    ("Market-moving disclosures (TSLA)", 'What did Tesla disclose in its most recent 8-K?', "TSLA", "8-K"),
    ("Big changes (NVDA)", 'Summarize Nvidia’s most recent 8-K.', "NVDA", "8-K"),
    ("Big changes (MSFT)", 'Summarize Microsoft’s most recent 8-K.', "MSFT", "8-K"),
]


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def extract_answer_and_first_source(output: str) -> str:
    # Keep only the Answer and the first source URL for cleaner briefs
    ans = ''
    src = ''
    m = re.search(r"=== Answer ===\n\n(.*?)(?:\n\n=== Sources|\Z)", output, re.S)
    if m:
        ans = m.group(1).strip()
    m2 = re.search(r"=== Sources \(top hits\) ===\n- (.+)", output)
    if m2:
        src = m2.group(1).strip()
    if not ans:
        ans = output.strip()
    if src:
        return ans + "\nSource: " + src
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default=os.getenv("TICKERS", TICKERS_DEFAULT))
    parser.add_argument("--forms", type=str, default="8-K")
    parser.add_argument("--limit", type=str, default="6")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    report_path = Path("reports") / f"daily_{today}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Scrape + clean
    run(["python", "scripts/scrape_sec.py", "--tickers", args.tickers, "--forms", args.forms, "--limit", args.limit])
    run(["python", "preprocess/clean_docs.py"])

    # 2) Chunk + update FAISS
    run(["python", "embeddings/chunk_docs.py"])
    run(["python", "embeddings/update_faiss.py"])

    # 3) Ask standard questions and write report
    lines = []
    lines.append(f"Daily Financial Brief — {today}\n")
    lines.append(f"Tickers: {args.tickers}\n")

    for title, q, ticker, doc_type in DAILY_QUESTIONS:
        lines.append("=" * 80)
        lines.append(title)
        lines.append("-" * 80)
        out = run([
            "python", "rag_pipeline/ask.py", q,
            "--ticker", ticker,
            "--doc-type", doc_type,
            "--most-recent",
            "--k", "80",
            "--max-chunks", "10"
        ])
        lines.append(extract_answer_and_first_source(out))
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
