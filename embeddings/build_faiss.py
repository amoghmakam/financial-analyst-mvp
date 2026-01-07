import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def load_chunks(jsonl_path: Path) -> List[dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str, batch_size: int = 64) -> np.ndarray:
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vecs = [d.embedding for d in resp.data]
        vectors.extend(batch_vecs)
    return np.array(vectors, dtype="float32")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="data/chunks/sec_chunks.jsonl")
    parser.add_argument("--out-dir", type=str, default="data/index")
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to .env")

    client = OpenAI(api_key=api_key)

    chunks_path = Path(args.chunks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    vectors = embed_texts(client, texts, model=args.model)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, str(out_dir / "sec.index"))

    with open(out_dir / "sec_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Saved FAISS index to {out_dir/'sec.index'}")
    print(f"Saved metadata to {out_dir/'sec_meta.json'}")


if __name__ == "__main__":
    main()
