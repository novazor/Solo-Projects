# test script for chunking process
from pathlib import Path
import json
import argparse
from typing import List, Dict
from src.chunk.text_chunker import iter_page_paragraphs, chunk_paragraphs

def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages_jsonl", type=Path)
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    pages = load_jsonl(args.pages_jsonl)
    paras = iter_page_paragraphs(pages)
    chunks = chunk_paragraphs(paras, target_chars=args.target, overlap_chars=args.overlap)

    out = Path("data/interim") / f"{args.pages_jsonl.stem.replace('_pages','')}_chunks.jsonl"
    save_jsonl(chunks, out)

    L = 150
    t0 = chunks[0]["text"][-L:]
    t1 = chunks[1]["text"][:L]
    print(t0 == t1) 
    print(repr(t0))
    print(repr(t1))


if __name__ == "__main__":
    main()