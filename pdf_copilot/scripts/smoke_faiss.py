# test script for facebook ai similarity search (FAISS)
from src.index_store.faiss_store import FaissStore
from src.embed.embedder import Embedder
from typing import List, Dict, Tuple 
import numpy as np
import json
from pathlib import Path

def read_jsonl(path: Path) -> tuple[List[str], List[Dict]]:
    texts = []
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            else:
                rec = json.loads(line)
                texts.append(rec["text"])
                metas.append({
                    "doc_id": rec["doc_id"],
                    "chunk_id": rec["chunk_id"],
                    "page_start": rec["page_start"],
                    "page_end": rec["page_end"],
                })
    return texts, metas
    
def main():
    json_path = Path('data/interim/input_chunks.jsonl')
    texts, metas = read_jsonl(json_path)
    index_path = Path('data/index/input.index')
    meta_path = Path('data/index/input_meta.json')
    e = Embedder()
    embeddings = e.embed_texts(texts)
    store = FaissStore(embeddings.shape[1], index_path, meta_path)
    store.build(embeddings, metas)
    assert store.index.ntotal == len(metas)

    query = "What is the submission deadline?"
    qvec = e.embed_texts([query])[0]
    hits = store.query(qvec, k=3)
    for hit in hits:
        idx, score = hit
        meta = store.metas[idx]
        snippet = texts[idx][:160]
        print(f'rank | {score:.3f} | {meta["chunk_id"]} | p{meta["page_start"]}-{meta["page_end"]} | {snippet}')
    

if __name__ == "__main__":
    main()