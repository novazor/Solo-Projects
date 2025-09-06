"""
retriever.py

Dense retrieval helpers. Embeds the query, asks the FAISS index for top-k,
and returns hits with text and metadata attached. Also includes a simple
score gate to decide when to answer vs. say “I don’t know.”

Exports:
- retrieve_topk(query, store, embedder, texts, k=6) -> list[dict]
  Each hit looks like: {"text": str, "score": float, ...meta_fields}
- has_enough_signal(hits, min_max: float, min_sum: float) -> bool

Conventions:
- Embeddings are normalized float32; FAISS uses inner product.
- `texts` aligns 1:1 with `store.metas` by index (0..N-1).
"""

from src.index_store.faiss_store import FaissStore
from src.embed.embedder import Embedder
from typing import List, Dict, Tuple 
import numpy as np
import json
from pathlib import Path

def retrieve_topk(query: str, store, embedder, texts: List[str], k: int = 6) -> List[Dict]:
    """
    Retrieves the top "k" results from searching the store with the embedded query
    TODO:
      - embed query (optionally prefix 'query: ')
      - store.query -> [(idx, score), ...]
      - return [{ "text": texts[idx], "score": score, **metas[idx] }]
    """
    q_vec = embedder.embed_texts([query])[0]
    hits = store.query(q_vec, k)
    output = []
    for hit in hits:
        idx, score = hit
        output.append({
                    "text": texts[idx],
                    "score": score, **store.metas[idx]})
    
    return output

def has_enough_signal(hits, min_max: float = 0.35, min_sum: float = 1.2) -> bool:
    """
    Return True if either the max similarity is strong enough or the sum of the top 3 is strong.
    Works even if some hits are missing/dirty scores.
    TODO:
      - Guard for cringe inputs
      - Loop through hits for similarity values
      - Return the top 3 (probably just sort list desc and return first 3)
    """
    if not hits:
        return False

    scores: list[float] = []
    for h in hits:
        try:
            scores.append(float(h.get("score", 0.0)))
        except Exception:
            continue

    if not scores:
        return False

    top3 = sorted(scores, reverse=True)[:3]
    return (top3[0] >= min_max) or (sum(top3) >= min_sum)

    