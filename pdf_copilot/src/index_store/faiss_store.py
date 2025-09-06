"""
faiss_store.py

Small wrapper around a FAISS index with a sidecar meta.json.
Holds file paths, builds or loads the index, and runs top-k searches.
Meta entries align 1:1 with vector ids (0..N-1), so you can recover doc/page info.

Exports:
- class FaissStore(dim, index_path, meta_path)
  .load()          # read index and metas from disk
  .query(vec, k=8) # return [(meta_idx, score), ...]
  # build/save helpers live here too in typical use.

Notes:
- Defaults assume inner-product on normalized embeddings; switch to L2 if needed.
- Keep the .index and .json together so reloads stay consistent.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
import json

# FaissStore class definition
class FaissStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path):
        """
        Keep paths; create index/metas later in build().
        """
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metas = None

    def build(self, embeddings, metas: List[Dict]) -> None:
        """
        Builds a FAISS index
        TODO:
          - Create FAISS index (IndexFlatIP or L2)
          - Add embeddings
          - Persist index and parallel metas (JSON)
        """
        assert embeddings.ndim == 2 and embeddings.dtype == np.float32
        assert embeddings.shape[1] == self.dim

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(embeddings))
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False)
        self.metas = metas

    def load(self):
        """Load the index + metas from disk."""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metas = json.load(f)

    def query(self, query_vec, k: int = 8) -> List[Tuple[int, float]]:
        """
        Return [(meta_index, score), ...] of top-k.
        """
        q = query_vec.astype(np.float32, copy=False)
        D, I = self.index.search(q[None, :], k)  
        hits = [(int(I[0, j]), float(D[0, j])) for j in range(I.shape[1])]
        return hits
