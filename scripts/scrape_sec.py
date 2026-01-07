import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv
from tqdm import tqdm

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{primary_doc}"
TICKER_TO_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_USER_AGENT = "FinancialAnalystMVP/0.1 (contact: you@example.com)"  # change later


def normalize_tickers(tickers: List[str]) -> List[str]:
    return [t.strip().upper() for t in tickers if t.strip()]


def sec_get(url: str, headers: Dict[str, str], timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def load_ticker_map(headers: Dict[str, str]) -> Dict[str, str]:
    r = sec_get(TICKER_TO_CIK_URL, headers=headers)
    data = r.json()
    mapping = {}
    for _, row in data.items():
        ticker = row.get("ticker", "").upper()
        cik_int = row.get("cik_str")
        if ticker and cik_int is not None:
            mapping[ticker] = str(cik_int).zfill(10)
    return mapping


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:200]


def build_filing_doc_url(cik10: str, accession: str, primary_doc: str) -> str:
    cik_nolead = str(int(cik10))  # remove leading zeros
    accession_nodash = accession.replace("-", "")
    return SEC_ARCHIVES_BASE.format(
        cik_nolead=cik_nolead,
        accession_nodash=accession_nodash,
        primary_doc=primary_doc,
    )


def scrape_sec_filings(
    tickers: List[str],
    out_dir: Path,
    form_types: List[str],
    limit_per_ticker: int,
    user_agent: str,
    sleep_s: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    ticker_map = load_ticker_map(headers)

    total_saved = 0
    for ticker in tqdm(tickers, desc="Tickers"):
        cik10 = ticker_map.get(ticker)
        if not cik10:
            print(f"[WARN] No CIK found for {ticker}. Skipping.")
            continue

        time.sleep(sleep_s)
        submissions = sec_get(SEC_SUBMISSIONS_URL.format(cik=cik10), headers=headers).json()

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        idxs = [i for i, f in enumerate(forms) if f in form_types][:limit_per_ticker]

        for i in idxs:
            form = forms[i]
            accession = accession_numbers[i]
            filing_date = filing_dates[i] if i < len(filing_dates) else None
            report_date = report_dates[i] if i < len(report_dates) else None
            primary_doc = primary_docs[i] if i < len(primary_docs) else None
            if not primary_doc:
                continue

            doc_url = build_filing_doc_url(cik10, accession, primary_doc)

            time.sleep(sleep_s)
            try:
                raw_html = sec_get(doc_url, headers=headers, timeout=30).text
            except Exception as e:
                print(f"[WARN] Failed doc {doc_url}: {e}")
                continue

            item = {
                "id": f"{ticker}_{accession}_{primary_doc}",
                "ticker": ticker,
                "source": "SEC_EDGAR",
                "doc_type": form,
                "cik": cik10,
                "accession": accession,
                "filing_date": filing_date,
                "report_date": report_date,
                "url": doc_url,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "raw_html": raw_html,
            }

            fpath = out_dir / (safe_filename(item["id"]) + ".json")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            total_saved += 1

    return total_saved


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default=os.getenv("TICKERS", "AAPL,MSFT,NVDA,TSLA,AMZN"))
    parser.add_argument("--out", type=str, default="data/raw/sec")
    parser.add_argument("--forms", type=str, default="8-K,10-Q")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--user-agent", type=str, default=os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT))
    args = parser.parse_args()

    tickers = normalize_tickers(args.tickers.split(","))
    forms = [f.strip() for f in args.forms.split(",") if f.strip()]

    saved = scrape_sec_filings(
        tickers=tickers,
        out_dir=Path(args.out),
        form_types=forms,
        limit_per_ticker=args.limit,
        user_agent=args.user_agent,
        sleep_s=args.sleep,
    )
    print(f"Saved {saved} SEC filings to {args.out}")


if __name__ == "__main__":
    main()
