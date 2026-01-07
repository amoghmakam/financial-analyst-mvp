import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def load_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def embed_texts(client: OpenAI, texts: List[str], model: str, batch_size: int = 64) -> np.ndarray:
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=str, default="data/chunks/sec_chunks.jsonl")
    p.add_argument("--index", type=str, default="data/index/sec.index")
    p.add_argument("--meta", type=str, default="data/index/sec_meta.json")
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    args = p.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    client = OpenAI(api_key=api_key)

    chunks_path = Path(args.chunks)
    index_path = Path(args.index)
    meta_path = Path(args.meta)

    chunks = load_jsonl(chunks_path)

    # If no index exists yet, fallback to full build
    if not index_path.exists() or not meta_path.exists():
        raise RuntimeError("Index/meta not found. Run embeddings/build_faiss.py first.")

    index = faiss.read_index(str(index_path))
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    existing_ids: Set[str] = set(x.get("chunk_id") for x in meta)
    new_items = [c for c in chunks if c.get("chunk_id") not in existing_ids]

    if not new_items:
        print("No new chunks to add. Index is up to date.")
        return

    texts = [x["text"] for x in new_items]
    vecs = embed_texts(client, texts, model=args.embed_model)

    index.add(vecs)
    meta.extend(new_items)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"Added {len(new_items)} new chunks. Index now has {index.ntotal} vectors.")


if __name__ == "__main__":
    main()
