"""
prompts.py

Central place for the app’s prompt text and small builders.
Holds the system/user templates for Q&A so model tweaks stay in one file.

Exports:
- build_context(hits: list[dict], max_chars: int = 3000) -> str
- qa_prompt(question: str, context: str) -> dict[str, str]

Conventions:
- Answers must include page tokens like [p3] or [p2-4] when supported by context.
- Be concise, ground claims in supplied text only, and say “I don't know.” when the answer isn’t in scope.

Placeholders used in templates:
{question}, {context}
"""

from typing import List, Dict
import re

# ------------------------------
# Light cleanup for chunk text
# ------------------------------
def sanitize_for_prompt(text: str, max_chars: int = 1000) -> str:
    """
    Normalize whitespace, fix hyphen-wrapped breaks, remove obvious page footers,
    and cap length to avoid blowing the context budget.
    """
    t = (text or "").replace("\r\n", "\n")
    t = re.sub(r"[ \t]+", " ", t)             # collapse spaces/tabs
    t = re.sub(r"\n{3,}", "\n\n", t)          # collapse blank lines
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)    # fix hyphen wrap
    t = re.sub(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", "", t, flags=re.I | re.M)  # "Page 3 of 12"
    t = t.strip()
    if len(t) > max_chars:
        cut = t[:max_chars]
        t = cut.rsplit(" ", 1)[0] + " …" if " " in cut else cut + " …"
    return t

# ------------------------------
# Small helpers for context pack
# ------------------------------
def _meta_get(hit: Dict, key: str):
    """Works for flattened metas OR nested under hit['metas']."""
    return hit.get(key, (hit.get("metas") or {}).get(key))

def _page_tag(ps: int, pe: int) -> str:
    return f"[p{ps}]" if ps == pe else f"[p{ps}-{pe}]"

def build_context(hits: List[Dict], max_chars: int = 3000) -> str:
    """
    Build a compact, readable context block from retrieval hits, ordered by score.
    Each block is headed by '--- [pX[-Y]] <chunk_id>' followed by cleaned text.
    """
    ordered = sorted(hits or [], key=lambda h: float(h.get("score", 0.0)), reverse=True)
    out, used = [], 0
    for h in ordered:
        ps, pe = _meta_get(h, "page_start"), _meta_get(h, "page_end")
        cid = _meta_get(h, "chunk_id") or "chunk"
        header = f"--- {_page_tag(ps, pe)} {cid}"
        body = sanitize_for_prompt(h.get("text", ""))
        block = header + "\n" + body
        sep = 2 if out else 0
        if used + sep + len(block) > max_chars:
            break
        if sep:
            out.append("")  # blank line separator
            used += sep
        out.append(block)
        used += len(block)
    return "\n".join(out)

def qa_prompt(question: str, context: str) -> Dict[str, str]:
    """
    Neutral Q&A prompt: strictly answer from Context with page citations.
    """
    system = (
        "You are a careful PDF analysis assistant. Answer ONLY from the provided Context. "
        "If the answer is not in the Context, respond exactly: \"I don't know.\" "
        "Cite pages strictly from the Context using [p3] or [p3-4]; do not invent citations or cite pages not present. "
        "Be concise (1–3 sentences)."
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations."
    return {"system": system, "user": user}

# ------------------------------
# smoke test 
# ------------------------------
if __name__ == "__main__":
    def _show(label: str, content: str) -> None:
        print(f"\n==== {label} ====\n{content}\n")

    # Sample hits
    hits = [
        {"text": "Abstract: We present a method that improves F1 by 7% on the benchmark dataset.",
         "score": 0.62, "chunk_id": "A-0003", "page_start": 3, "page_end": 3},
        {"text": "Conclusion: Future work includes multi-modal fusion and larger-scale evaluation.",
         "score": 0.48, "chunk_id": "A-0007", "page_start": 7, "page_end": 8},
        {"text": "References include Smith et al. (2021) and Lee & Gomez (2024).",
         "score": 0.35, "chunk_id": "B-0002", "page_start": 2, "page_end": 2},
    ]

    ctx = build_context(hits)
    msgs = qa_prompt("What improvement did the method report?", ctx)

    _show("Context", ctx)
    _show("System", msgs["system"])
    print("==== User ====\n", msgs["user"][:600], "...\n")

    # Quick checks
    assert "ONLY" in msgs["system"] and "I don't know" in msgs["system"], "System guardrails missing"
    assert "[p3]" in ctx and "[p7-8]" in ctx and "[p2]" in ctx, "Missing page tags in context"
    assert "Question:" in msgs["user"] and "Context:" in msgs["user"], "User message shape off"
    print("✅ qa_prompt smoke passed.")
