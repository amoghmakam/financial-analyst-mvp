import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


def embed_query(client: OpenAI, q: str, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[q])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--index", type=str, default="data/index/sec.index")
    parser.add_argument("--meta", type=str, default="data/index/sec_meta.json")
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--chat-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--k", type=int, default=25, help="retrieve more then filter")
    parser.add_argument("--ticker", type=str, default=None, help="e.g., AMZN")
    parser.add_argument("--doc-type", type=str, default=None, help="e.g., 8-K")
    parser.add_argument("--most-recent", action="store_true", help="pick most recent doc after filtering")
    parser.add_argument("--max-chunks", type=int, default=8, help="chunks to include in context")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    client = OpenAI(api_key=api_key)

    index = faiss.read_index(args.index)
    meta = json.load(open(args.meta, "r", encoding="utf-8"))

    qvec = embed_query(client, args.question, args.embed_model)
    scores, idxs = index.search(qvec, args.k)

    # Collect hits
    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0:
            continue
        item = meta[idx]
        m = item.get("meta", {})
        hits.append((score, item, m))

    # Optional metadata filter
    if args.ticker:
        t = args.ticker.strip().upper()
        hits = [h for h in hits if (h[2].get("ticker") or "").upper() == t]

    if args.doc_type:
        dt = args.doc_type.strip().upper()
        hits = [h for h in hits if (h[2].get("doc_type") or "").upper() == dt]

    if not hits:
        print("\n=== Answer ===\n")
        print("No matching documents found after filtering.")
        return

    # If most-recent, pick the doc (by URL) with latest filing_date
    if args.most_recent:
        by_url = defaultdict(list)
        for score, item, m in hits:
            by_url[m.get("url")].append((score, item, m))

        best_url = None
        best_date = None
        for url, items in by_url.items():
            # use max filing date among chunks for that url
            dates = [parse_date(x[2].get("filing_date")) for x in items]
            dates = [d for d in dates if d]
            dmax = max(dates) if dates else None
            if dmax and (best_date is None or dmax > best_date):
                best_date = dmax
                best_url = url

        if best_url:
            hits = by_url[best_url]

    # Sort hits by similarity score and take top chunks
    hits = sorted(hits, key=lambda x: x[0], reverse=True)[: args.max_chunks]

    context_blocks = []
    sources = []
    for score, item, m in hits:
        context_blocks.append(
            f"[{m.get('ticker')} | {m.get('doc_type')} | {m.get('filing_date')}] {item.get('text','')}"
        )
        if m.get("url"):
            sources.append(m["url"])

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a financial market analyst assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say what is missing (e.g., "the filing text doesn't include the disclosure section").

Question: {args.question}

Context:
{context}
"""

    resp = client.chat.completions.create(
        model=args.chat_model,
        messages=[
            {"role": "system", "content": "Be concise. Use bullet points. Prefer concrete facts with dates."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    print("\n=== Answer ===\n")
    print(resp.choices[0].message.content)

    print("\n=== Sources (top hits) ===")
    for s in list(dict.fromkeys([u for u in sources if u]))[:5]:
        print("-", s)


if __name__ == "__main__":
    main()
