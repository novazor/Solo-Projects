"""
text_chunker.py

This file handles turning PDF text into chunks for retrieval.
Keeps doc_id and page span so citations are stil accurate later.
"""
from typing import List, Dict, Iterable
import re

def iter_page_paragraphs(pages: List[Dict]) -> Iterable[Dict]:
    """
    Iterates through page paragraphs, parsed with Pymupdf
    """
    for page in pages:
        full_text = page["text"]
        paragraphs = re.split(r"[\r\n]{2,}", full_text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) < 40 and not para.istitle() or para.isupper():
                continue
            yield {
                    "doc_id": page["doc_id"],
                    "page_num": page["page_num"],
                    "paragraph": para,
                }

def chunk_paragraphs(
    para_stream: Iterable[Dict],
    target_chars: int = 1200,
    overlap_chars: int = 150
) -> List[Dict]:
    """
    Build chunks that are roughly ~target_chars with overlap.
    Keep metadata:
      - doc_id (single doc here)
      - chunk_id (e.g., f"{doc_id}-{i:04d}")
      - page_start, page_end
      - text
    Basically it'll do this:
      - Keep a buffer (string) and current page span (min/max page).
      - Append paragraphs until buffer >= target; emit chunk.
      - For overlap: take last overlap_chars from emitted chunk to use for the start of next buffer.
    """
    output = []
    buffer = []
    buf_len = 0
    page_start = None
    page_end = None
    i = 0

    for para in para_stream:
      ptxt = para["paragraph"]

      buffer.append(ptxt)
      buf_len += len(ptxt) + 2
      page_start = para["page_num"] if page_start is None else min(page_start, para["page_num"])
      page_end   = para["page_num"] if page_end   is None else max(page_end,   para["page_num"])

      if buf_len >= target_chars:
          text = "\n\n".join(buffer)
          output.append({
              "doc_id": para["doc_id"],
              "chunk_id": f'{para["doc_id"]}-{i:04d}',
              "page_start": page_start,
              "page_end": page_end,
              "text": text
          })
          i += 1
          tail_len = min(overlap_chars, len(text))
          tail = text[-tail_len:]
          buffer = [tail]
          buf_len = tail_len
          page_start = page_end  # reasonable guess for the next span seed
    # flush
    if buffer and buf_len > 40:
        text = "\n\n".join(buffer)
        output.append({
            "doc_id": para["doc_id"],
              "chunk_id": f'{para["doc_id"]}-{i:04d}',
              "page_start": page_start,
              "page_end": page_end,
              "text": text
          })
    return output

if __name__ == "__main__":
    print("Dev check only — run scripts.make_chunks for real CLI.")
