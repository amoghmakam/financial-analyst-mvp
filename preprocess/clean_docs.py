import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_one(raw_item: Dict) -> Optional[Dict]:
    raw_html = raw_item.get("raw_html")
    if not raw_html:
        return None

    text = html_to_text(raw_html)
    if len(text) < 200:
        return None

    return {
        "id": raw_item.get("id"),
        "text": text,
        "ticker": raw_item.get("ticker"),
        "source": raw_item.get("source"),
        "doc_type": raw_item.get("doc_type"),
        "cik": raw_item.get("cik"),
        "accession": raw_item.get("accession"),
        "filing_date": raw_item.get("filing_date"),
        "report_date": raw_item.get("report_date"),
        "url": raw_item.get("url"),
        "fetched_at": raw_item.get("fetched_at"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="data/raw/sec")
    parser.add_argument("--out-dir", type=str, default="data/clean/sec")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        print(f"No files in {in_dir}. Run the scraper first.")
        return

    saved = 0
    for fp in tqdm(files, desc="Cleaning"):
        with open(fp, "r", encoding="utf-8") as f:
            raw_item = json.load(f)
        cleaned = clean_one(raw_item)
        if not cleaned:
            continue
        out_fp = out_dir / fp.name
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        saved += 1

    print(f"Cleaned {saved} docs to {out_dir}")


if __name__ == "__main__":
    main()
