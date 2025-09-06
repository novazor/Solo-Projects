"""
answerer.py

Thin wrapper that turns retrieved chunks into a final answer.
Runs a map/reduce prompt over hits, extracts page citations like [p3] or [p2-4],
and returns both the answer text and which chunks were used.

Exports:
- answer_question(store, embedder, llm, question, texts, k=6) -> {answer, citations, used}
- summarize_document(texts, metas, llm, style="Bullet Points", target_units=8) -> {...}
- extract_citations(text) -> list[str]
- preview_used(hits, n=3) -> list[str]  # debug/printing helper

Notes:
- Uses dense retrieval (retrieve_topk) upstream of the LLM call.
- Domain-agnostic (no procurement heuristics).
- Citations are parsed from the model’s output and normalized to page tokens.
"""

from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path
import json
import re

from src.index_store.faiss_store import FaissStore  
from src.embed.embedder import Embedder             
from src.retrieve.retriever import retrieve_topk, has_enough_signal
from src.generate.prompts import build_context, qa_prompt
from src.generate.llm_client import OllamaClient   


# --------------------------------------------------------
# Citations
# --------------------------------------------------------

def _token_pages(tok: str) -> set[int]:
    if "-" in tok or "–" in tok:
        a, b = re.split(r"[-–]", tok)
        return set(range(int(a), int(b) + 1))
    return {int(tok)}

def extract_citations(text: str) -> list[str]:
    """
    - Regex: r"\[p(\d+)(?:-(\d+))?\]"
    - Convert matches to "X" or "X-Y"
    - Deduplicate while preserving first occurrence order
    - Return the list
    """
    output, seen = [], set()
    pattern = r"\[p\s*(\d+)\s*(?:[-–]\s*(\d+))?\]"
    matches = re.finditer(pattern, text or "", flags=re.I)
    for match in matches:
        a = str(int(match.group(1)))
        b = match.group(2)
        token = a if b is None or a == str(int(b)) else f"{a}-{int(b)}"
        if token not in seen:
            output.append(token)
            seen.add(token)
    return output


# --------------------------------------------------------
# Generic PDF Summarizer (map/reduce)
# --------------------------------------------------------

_MAP_MAX_TOKENS = 280
_REDUCE_MAX_TOKENS = 700
_TEMPERATURE = 0.2

def _llm_call(llm, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Small adapter to support different llm client shapes:
    - ollama-like: llm.generate(prompt=..., max_tokens=..., temperature=...)
    - openai-like wrapper: llm.complete(prompt, max_tokens=..., temperature=...)
    - bare callable: llm(prompt)
    """
    if hasattr(llm, "generate"):
        return (llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature) or "").strip()
    if hasattr(llm, "complete"):
        return (llm.complete(prompt, max_tokens=max_tokens, temperature=temperature) or "").strip()
    try:
        return (llm(prompt) or "").strip()
    except Exception:
        return ""

def _prompt_map_chunk(style: str) -> str:
    """
    Instruction for summarizing ONE chunk.
    Neutral and requires the provided page tag in the output.
    """
    style_hint = {
        "Bullet Points": "Write 1–2 concise bullet points.",
        "Narrative": "Write 1–2 concise sentences.",
        "Outline": "Write 1–2 concise outline items.",
    }.get(style, "Write 1–2 concise bullet points.")

    return (
        "You are a careful, neutral PDF Summarizer.\n"
        f"{style_hint} Only include facts that appear in the text. Do not guess.\n"
        "Include the page tag I provide (e.g., [p3] or [p3–4]) in each item so citations are preserved.\n"
        "Do not add extra commentary. Keep it tight and faithful."
    )

def _prompt_reduce(style: str, target_units: int) -> str:
    if style == "Bullet Points":
        format_hint = f"Return ~{target_units} crisp bullet points."
    elif style == "Narrative":
        format_hint = f"Return ~{target_units} short sentences assembled into 1–2 tight paragraphs."
    else:  # Outline
        format_hint = f"Return ~{target_units} outline sections with short subpoints as needed."

    return (
        "Merge the following items into a concise, de-duplicated summary of the document.\n"
        f"{format_hint}\n"
        "Only keep information present in the items. Preserve any page tags like [p5] or [p6–7] so citations remain visible.\n"
        "Remove repetitions, merge near-duplicates, use clear wording, and keep it factual."
    )

def _build_page_tag(meta: Dict) -> str:
    ps = meta.get("page_start")
    pe = meta.get("page_end")
    if ps is None and pe is None:
        return "[p?]"
    if pe is None or pe == ps:
        return f"[p{ps}]"
    # en dash for aesthetics; parser accepts '-' and '–'
    return f"[p{ps}–{pe}]"

def _dedup_lines(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in lines:
        for ln in str(raw).splitlines():
            s = ln.strip()
            if not s:
                continue
            k = re.sub(r"\s+", " ", s.casefold())
            if k not in seen:
                seen.add(k)
                out.append(s)
    return out

def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]

_TAG_RX = re.compile(r"\[p(\d+)(?:[–-](\d+))?\]")

def _extract_and_merge_page_tags(text: str) -> List[str]:
    pairs = []
    for m in _TAG_RX.finditer(text or ""):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            a, b = b, a
        pairs.append((a, b))
    if not pairs:
        return []
    points = set()
    for a, b in pairs:
        for x in range(a, b + 1):
            points.add(x)
    if not points:
        return []
    sorted_pts = sorted(points)
    ranges = []
    start = prev = sorted_pts[0]
    for x in sorted_pts[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    tags = []
    for a, b in ranges:
        tags.append(f"[p{a}]" if a == b else f"[p{a}–{b}]")
    return tags

def _iterative_reduce(items: List[str], style: str, target_units: int, llm) -> str:
    curr = _dedup_lines(items)
    if not curr:
        return ""

    if len(curr) <= max(8, target_units * 2):
        prompt = _prompt_reduce(style, target_units)
        joined = "\n".join(f"- {s}" for s in curr)
        return _llm_call(llm, f"{prompt}\n\nItems:\n{joined}", max_tokens=_REDUCE_MAX_TOKENS, temperature=_TEMPERATURE)

    batch_size = 24
    batches = _chunk_list(curr, batch_size)
    merged_batches: List[str] = []
    for batch in batches:
        prompt = _prompt_reduce(style, max(target_units, len(batch)//2))
        joined = "\n".join(f"- {s}" for s in batch)
        merged = _llm_call(llm, f"{prompt}\n\nItems:\n{joined}", max_tokens=_REDUCE_MAX_TOKENS, temperature=_TEMPERATURE)
        if merged:
            merged_batches.extend(_dedup_lines(merged.splitlines()))

    prompt_final = _prompt_reduce(style, target_units)
    joined_final = "\n".join(f"- {s}" for s in merged_batches)
    return _llm_call(llm, f"{prompt_final}\n\nItems:\n{joined_final}", max_tokens=_REDUCE_MAX_TOKENS, temperature=_TEMPERATURE)

def summarize_document(
    texts: List[str],
    metas: List[Dict],
    llm,
    style: str = "Bullet Points",
    target_units: int = 8,
) -> Dict[str, object]:
    """
    Map-Reduce summarizer over ALL chunks (domain-agnostic).
    Returns: {'answer': str, 'citations': [str], 'used': []}
    """
    if not texts or not metas or len(texts) != len(metas):
        return {"answer": "No content indexed. Please (re)build the index.", "citations": [], "used": []}

    micros: List[str] = []
    map_instr = _prompt_map_chunk(style)
    for text, meta in zip(texts, metas):
        if not text or not str(text).strip():
            continue
        tag = _build_page_tag(meta)
        prompt = f"{map_instr}\n\nPage tag: {tag}\n\nChunk:\n{text}"
        micro = _llm_call(llm, prompt, max_tokens=_MAP_MAX_TOKENS, temperature=_TEMPERATURE)
        if not micro:
            continue
        fixed_lines = []
        for ln in micro.splitlines():
            s = ln.strip()
            if not s:
                continue
            if tag not in s:
                s = f"{tag} {s.lstrip('- ').strip()}"
            fixed_lines.append(s)
        micros.extend(fixed_lines)

    micros = _dedup_lines(micros)
    if not micros:
        return {"answer": "Could not extract a summary from the provided pages.", "citations": [], "used": []}

    merged = _iterative_reduce(micros, style, target_units, llm).strip()
    if not merged:
        return {"answer": "Summarization step returned no content.", "citations": [], "used": []}

    citations = _extract_and_merge_page_tags(merged)

    if style == "Bullet Points":
        if not re.search(r"^\s*[-•]\s", merged, flags=re.M):
            merged = "\n".join(f"- {ln.strip()}" for ln in merged.splitlines() if ln.strip())
    elif style == "Outline":
        lines = [ln.strip() for ln in merged.splitlines() if ln.strip()]
        if not any(re.match(r"^\d+[\.\)]\s", ln) for ln in lines):
            merged = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))

    return {"answer": merged, "citations": citations, "used": []}


# --------------------------------------------------------
# Q&A 
# --------------------------------------------------------

def answer_question(store, embedder, llm, question: str, texts: List[str], k: int = 6) -> Dict:
    """
    Dense retrieval → context pack → QA prompt → citation check.
    No procurement/bond heuristics.
    """
    # 1) Retrieval
    hits = retrieve_topk(query=question, store=store, embedder=embedder, texts=texts, k=k)

    # 2) Retrieval gate (kept behavior, slightly different thresholds for MiniLM)
    model_name = getattr(embedder, "name", "") or getattr(getattr(embedder, "model", None), "name", "")
    if "minilm" in str(model_name).lower():
        ok = has_enough_signal(hits, min_max=0.30, min_sum=0.90)
    else:
        ok = has_enough_signal(hits, min_max=0.35, min_sum=1.20)

    if not ok:
        return {"answer": "I don't know.", "citations": [], "used": hits[:k]}

    # 3) Context + prompt
    ctx = build_context(hits[:k])
    msgs = qa_prompt(question, ctx)

    # 4) Generate
    raw = llm.generate(system=msgs["system"], user=msgs["user"], max_tokens=256, temperature=0.1)

    # 5) Extract cites
    cites = extract_citations(raw)

    # 6) Guard: only allow pages present in the context
    allowed_pages: set[int] = set()
    for h in hits[:k]:
        try:
            ps, pe = int(h["page_start"]), int(h["page_end"])
            allowed_pages.update(range(ps, pe + 1))
        except (KeyError, TypeError, ValueError):
            continue

    invalid = [t for t in cites if not _token_pages(t).issubset(allowed_pages)]
    if (not (raw or "").strip()) or invalid:
        return {"answer": "I don't know.", "citations": [], "used": hits[:k]}

    return {"answer": raw.strip(), "citations": cites, "used": hits[:k]}


# --------------------------------------------------------
# Debug helpers
# --------------------------------------------------------

def preview_used(hits: List[Dict], n: int = 3) -> List[str]:
    """
    Helper for printing: return strings like
    "A-0003  [p3]    score=0.623" (respect pX vs pX-Y).

    - Robustly sorts hits by score (descending), even if score is a string or missing.
    - Safely converts page_start/page_end/score; falls back to placeholders on bad data.
    """
    def to_int(v) -> Optional[int]:
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None

    def to_float(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("-inf")  # pushes bad/missing scores to the end

    ordered = sorted(hits or [], key=lambda h: to_float(h.get("score")), reverse=True)

    lines: List[str] = []
    for h in ordered[:n]:
        pgs = to_int(h.get("page_start"))
        pge = to_int(h.get("page_end"))
        score = to_float(h.get("score"))

        if pgs is not None and pge is not None:
            page_tag = f"[p{pgs}]" if pgs == pge else f"[p{pgs}-{pge}]"
        else:
            page_tag = "[p?]"

        chunk_id = str(h.get("chunk_id", "unknown"))
        score_str = "n/a" if score == float("-inf") else f"{score:.3f}"

        lines.append(f"{chunk_id}  {page_tag}  score={score_str}")

    return lines


# --------------------------------------------------------
# Smoke test
# --------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke-test both preview_used(...) and answer_question(...).

    Run from project root (venv active):
        python -m src.generate.answerer
    """

    print("== preview_used(sample_hits) ==")
    sample_hits = [
        {"chunk_id": "A-0003", "page_start": 3, "page_end": 3, "score": 0.623,
         "text": "Submit via the City Portal by February 18, 2025 at 2:00 PM ET."},
        {"chunk_id": "A-0007", "page_start": 7, "page_end": 8, "score": 0.481,
         "text": "No performance bond is required for this solicitation."},
        {"chunk_id": "B-0002", "page_start": 2, "page_end": 2, "score": 0.352,
         "text": "Questions due by February 7, 2025 at 5:00 PM ET."},
    ]
    for line in preview_used(sample_hits, n=3):
        print("  ", line)

    chunks_path = Path("data/interim/input_chunks.jsonl")
    index_path  = Path("data/index/input.index")
    meta_path   = Path("data/index/input_meta.json")

    if not (chunks_path.exists() and index_path.exists() and meta_path.exists()):
        print("\n[skip] E2E QA: missing artifacts. Expected:")
        print(f"  - {chunks_path}")
        print(f"  - {index_path}")
        print(f"  - {meta_path}")
    else:
        texts: List[str] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    texts.append(obj["text"])
                except Exception:
                    continue

        embedder = Embedder()
        try:
            dim = embedder.dim
        except AttributeError:
            dim = embedder.embed_texts(["_"]).shape[1]

        try:
            store = FaissStore(dim=dim, index_path=index_path, meta_path=meta_path)
            store.load()
        except Exception as e:
            print("\n[error] Failed to load FaissStore:", repr(e))
            store = None

        if store is None:
            print("\n[skip] E2E QA: could not initialize FAISS store.")
        else:
            llm = OllamaClient(model="mistral:instruct")

            questions = [
                "What is the submission deadline?",
                "Where do I submit the proposal?",
                "Is a performance bond required?",
                "What color is the mayor’s car?",
            ]

            for q in questions:
                print("\nQ:", q)
                try:
                    out = answer_question(store, embedder, llm, q, texts, k=6)
                except Exception as e:
                    print("  [error] answer_question failed:", repr(e))
                    continue

                ans = (out.get("answer") or "").strip()
                lines = [ln for ln in ans.splitlines() if ln.strip()]
                print("A:", "\n   ".join(lines[:3]) if lines else "<empty>")
                print("Citations:", out.get("citations", []))
                print("Used (top-3):")
                for line in preview_used(out.get("used", []), n=3):
                    print("  ", line)

            print("\n✅ Smoke complete. If answers truncate, bump max_tokens in answer_question().")
