import argparse
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="data/clean/sec")
    parser.add_argument("--out", type=str, default="data/chunks/sec_chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        print(f"No cleaned docs found in {in_dir}")
        return

    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for fp in tqdm(files, desc="Chunking"):
            doc = json.load(open(fp, "r", encoding="utf-8"))
            text = doc.get("text", "")
            if not text:
                continue

            chunks = chunk_text(text, args.chunk_size, args.overlap)
            for idx, ch in enumerate(chunks):
                item = {
                    "chunk_id": f"{doc.get('id')}::chunk_{idx}",
                    "text": ch,
                    "meta": {
                        "id": doc.get("id"),
                        "ticker": doc.get("ticker"),
                        "doc_type": doc.get("doc_type"),
                        "filing_date": doc.get("filing_date"),
                        "url": doc.get("url"),
                        "source": doc.get("source"),
                    },
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} chunks to {out_path}")


if __name__ == "__main__":
    main()
