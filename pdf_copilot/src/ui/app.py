"""
app.py

Streamlit front end for PDF Copilot (RAG over PDFs).
Upload PDFs, turn pages into chunks, build or load a FAISS index,
then ask questions with page-linked citations.

Main pieces:
- Sidebar: model tag, top_k, chunk size/overlap, rebuild toggle, index path.
- Tabs: Upload, Make Chunks, Build/Load Index, Ask.
- Pipeline hooks: pdf_loader.load_pdf -> text_chunker.chunk_paragraphs
  -> faiss_store.FaissStore (build/load/query) -> retriever.retrieve_topk
  -> answerer.answer_question.

Session state:
- Stores pages, chunks, texts/metas, index_dir, last hits, and settings.

Run:
- streamlit run src/ui/app.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List, Optional, Tuple
import streamlit as st
import numpy as np
import os
import shutil, uuid
import json
import re
from datetime import datetime
from src.ingest.pdf_loader import *
from src.generate.llm_client import OllamaClient
from src.generate.answerer import *
from src.chunk.text_chunker import *
from src.embed.embedder import *
from src.index_store.faiss_store import *
import hashlib

AppState = Dict[str, object]

def get_state() -> AppState:
    """
    Returns a mutable state bag for the app.
    No input parameters.
    Populates/holds keys like:
      - 'embedder', 'llm', 'store', 'texts', 'metas'
      - 'last_hits', 'last_answer', 'last_question'
      - 'paths': {'interim_dir': Path, 'index_dir': Path}
    """
    if 'app_state' not in st.session_state:
        temp = {
           'embedder': None, 'llm': None, 'store': None,
           'texts': None, 'metas': None,
           'last_hits': None, 'last_answer': None, 'last_question': None,
           'paths': {'interim_dir': Path('data/interim'), 'index_dir': Path('data/index')}
         }
        st.session_state['app_state'] = temp

    return st.session_state['app_state']

# --- Session-scoped folders (one per app run) -------------------
def _ensure_session_dirs(state: dict, interim_root: Path, index_root: Path) -> tuple[Path, Path]:
    """
    Creates fresh session folders: data/interim/session_<id>, data/index/session_<id>.
    On first call, deletes any old session_* folders so we start clean.
    Returns (interim_session_dir, index_session_dir).
    """
    state.setdefault("paths", {})
    if not state["paths"].get("session_ready"):
        # wipe old sessions
        for root in (interim_root, index_root):
            if root.exists():
                for p in root.glob("session_*"):
                    shutil.rmtree(p, ignore_errors=True)

        sid = f"session_{uuid.uuid4().hex[:8]}"
        state["paths"]["session_id"] = sid
        state["paths"]["interim_session"] = str((interim_root / sid).resolve())
        state["paths"]["index_session"] = str((index_root / sid).resolve())
        Path(state["paths"]["interim_session"]).mkdir(parents=True, exist_ok=True)
        Path(state["paths"]["index_session"]).mkdir(parents=True, exist_ok=True)
        state["paths"]["session_ready"] = True

    return Path(state["paths"]["interim_session"]), Path(state["paths"]["index_session"])
# ----------------------------------------------------------------

def render_upload_panel(
    state: AppState,
    interim_dir: Path,
    allow_multiple: bool = True,
    accepted_types: Tuple[str, ...] = (".pdf",),
) -> None:
    """
    Renders the file uploader UI (SESSION-SCOPED).

    Behavior change:
      • Each new upload batch wipes any *.pdf / *.jsonl already in this session folder.
      • Writes one <stem>_pages.jsonl per uploaded PDF.
      • Replaces state['pages_files'] with only this batch's jsonls.

    Side effects:
      - Writes artifacts under the *session* interim_dir.
      - Updates state['paths'] and state['pages_files'] (REPLACED, not extended).
    """
    # -------------------- helpers --------------------
    def _sanitize_stem(original_name: str) -> str:
        temp = Path(original_name).stem
        temp = temp.strip().lower().replace(" ", "_")
        temp = re.sub(r"[^a-z0-9_]+", "", temp)[:80]
        return temp or "upload"

    def _persist_uploaded_files(uploaded_files, _interim_dir: Path) -> list[Path]:
        _interim_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for idx, uf in enumerate(uploaded_files):
            stem = _sanitize_stem(uf.name)
            dest = _interim_dir / f"{stem}-{idx}.pdf"
            with dest.open("wb") as out:
                out.write(uf.read())
            paths.append(dest)
        return paths

    def _pdf_to_pages(pdf_path: Path) -> list[dict]:
        return load_pdf(pdf_path)

    def _write_jsonl(records: list[dict], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # ---------------- end helpers --------------------

    types = [ext.lstrip(".") for ext in accepted_types]
    files = st.file_uploader("Upload PDFs", type=types, accept_multiple_files=allow_multiple)

    if files is None:
        st.info("Upload one or more PDFs to begin.")
        return
    if not isinstance(files, list):
        files = [files]
    if len(files) == 0:
        st.info("No files selected.")
        return

    # session reset - wipes prior PDFs/JSONLs in this session folder
    with st.spinner("Preparing session folder…"):
        interim_dir.mkdir(parents=True, exist_ok=True)
        for p in interim_dir.glob("*"):
            if p.suffix.lower() in {".pdf", ".jsonl"}:
                try:
                    p.unlink()
                except Exception:
                    pass

    summary_rows: list[dict] = []
    jsonl_paths: list[Path] = []

    with st.spinner("Processing uploads…"):
        pdf_paths = _persist_uploaded_files(files, interim_dir)
        for pdf_path in pdf_paths:
            try:
                pages = _pdf_to_pages(pdf_path)
                out_jsonl = interim_dir / f"{pdf_path.stem}_pages.jsonl"
                _write_jsonl(pages, out_jsonl)
                jsonl_paths.append(out_jsonl)
                summary_rows.append({"pdf": pdf_path.name, "pages": len(pages), "jsonl": out_jsonl.name})
            except Exception as e:
                summary_rows.append({"pdf": pdf_path.name, "error": str(e)})

    # Replace (not extend) session state to track only this batch
    state.setdefault("paths", {})
    state["paths"]["interim_dir"] = interim_dir
    state["pages_files"] = [str(p) for p in jsonl_paths]
    state["pages_info"] = summary_rows

    st.success(f"Processed {len(jsonl_paths)} file(s).")
    st.dataframe(summary_rows, use_container_width=True)
    st.info("Next: open the **Index** tab to chunk & embed. "
            "Note: the Index tab uses only this session’s upload(s).")


def render_index_panel(
    state: AppState,
    interim_dir: Path,
    index_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    rebuild: bool = True,
) -> None:
    """
    Build chunks from *_pages.jsonl and create/load a FAISS index.
    """

    # ---------- Helpers ----------
    def _discover_pages_files(interim_dir: Path, state: dict) -> list[Path]:
        """Return only this session’s *_pages.jsonl files."""
        files = sorted(Path(interim_dir).glob("*_pages.jsonl"))
        # keep state in sync (string paths for serialization)
        state["pages_files"] = [str(p) for p in files]
        return files

    def _make_chunks_for(pages_jsonl: Path) -> Optional[Path]:
        lines_list: List[dict] = []
        try:
            with pages_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines_list.append(json.loads(line))
        except FileNotFoundError:
            st.error(f"Missing pages file: {pages_jsonl}")
            return None

        paras = iter_page_paragraphs(lines_list)
        chunks = chunk_paragraphs(paras)

        out_jsonl = _chunks_path_for(pages_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for entry in chunks:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return out_jsonl

    def _chunks_path_for(pages_jsonl: Path) -> Path:
        """
        Given '..._pages.jsonl' return '..._chunks.jsonl'.
        Works even if the stem doesn't end with '_pages'.
        """
        stem = pages_jsonl.stem
        if stem.endswith("_pages"):
            stem = stem[:-6]  # drop the suffix '_pages'
        return pages_jsonl.with_name(f"{stem}_chunks.jsonl")

    # ---------- Panel body ----------
    pages_files = _discover_pages_files(interim_dir, state)
    if not pages_files:
        st.info("Upload a PDF in the Upload tab to create a pages.jsonl for this session.")
        return

    # Auto-select ONLY the latest file in this session
    latest = max(pages_files, key=lambda p: p.stat().st_mtime)
    selected_pages = [latest]
    st.caption(f"Using pages.jsonl: `{latest.name}`")

        
    # Per-corpus index directory based on selection + model
    corpus_id = _make_corpus_id(selected_pages, model_name)
    active_index_dir = Path(index_dir) / corpus_id
    st.caption(f"Corpus: `{corpus_id}` → {active_index_dir.as_posix()}")

    # Make Chunks
    if st.button("Make Chunks", disabled=not selected_pages, key="btn_make_chunks"):
        rows, produced = [], []
        with st.spinner("Chunking pages…"):
            for p in selected_pages:
                try:
                    out_path = _make_chunks_for(p)
                    if out_path is None:
                        rows.append({"pages": p.name, "chunks": "—", "num_chunks": 0, "status": "error"})
                        continue
                    # count chunks
                    n = 0
                    with out_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                n += 1
                    rows.append({"pages": p.name, "chunks": out_path.name, "num_chunks": n, "status": "ok"})
                    produced.append(out_path.resolve())
                except Exception as e:
                    rows.append({"pages": p.name, "chunks": "—", "num_chunks": 0, "status": str(e)})
        if produced:
            state.setdefault("chunk_files", [])
            # de-dup
            already = {str(Path(x)) for x in state["chunk_files"]}
            for c in produced:
                if str(c) not in already:
                    state["chunk_files"].append(c)
        st.success(f"Chunked {sum(1 for r in rows if r['status']=='ok')}/{len(rows)} file(s).")
        st.dataframe(rows, use_container_width=True)

    # Build / Load Index
    if st.button("Build / Load Index", disabled=not selected_pages, key="btn_build_index"):
        # Decide which chunks to use (auto-create if missing)
        chunks_to_use: List[Path] = []

        # Prefer deterministic selection from the currently selected pages.jsonl
        for p in selected_pages:
            cpath = _chunks_path_for(p)
            if not cpath.exists():
                # auto-make chunks on the fly
                made = _make_chunks_for(p)
                if made:
                    cpath = Path(made)
            if cpath.exists():
                chunks_to_use.append(cpath.resolve())

        # (Optional) also include any session chunk files the user made earlier
        for p in (state.get("chunk_files") or []):
            try:
                pp = Path(p)
                if pp.exists() and pp.resolve() not in chunks_to_use:
                    chunks_to_use.append(pp.resolve())
            except Exception:
                pass

        if not chunks_to_use:
            st.info("No chunks found—click **Make Chunks** first or upload again.")
            return


        texts_all: List[str] = []
        metas_all: List[Dict] = []
        with st.spinner("Loading chunks…"):
            for cpath in chunks_to_use:
                t, m = _read_chunks_jsonl(cpath)
                if t:
                    texts_all.extend(t)
                    metas_all.extend(m)

        if not texts_all:
            st.warning("No valid chunks loaded.")
            return

        embedder = _ensure_embedder(state, model_name)
        with st.spinner("Embedding & building index…"):
            # Build/load under a per-corpus directory (prevents cross-run contamination)
            store, stats = _build_or_load_faiss(
                texts_all, metas_all, active_index_dir, embedder, rebuild=rebuild
            )

        # Fresh session state for this corpus
        _reset_index_state(state)

        # Augment stats and commit new corpus into state
        stats["meta_path"] = str(Path(active_index_dir) / "meta.json")
        state["texts"] = texts_all
        state["metas"] = metas_all
        state["store"] = store
        state.setdefault("paths", {})["index_dir"] = active_index_dir
        state["index_stats"] = stats
        state["embed_model_name"] = model_name

        st.success("Index ready.")
        _index_stats_ui(stats)


def _read_chunks_jsonl(chunks_path: Path) -> Tuple[List[str], List[Dict]]:
    """Load chunks.jsonl → (texts, metas) preserving order."""
    texts: List[str] = []
    metas: List[Dict] = []

    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Be flexible on the text field name
                txt = (
                    rec.get("text")
                    or rec.get("chunk_text")
                    or rec.get("content")
                    or rec.get("body")
                )
                if not isinstance(txt, str) or not txt.strip():
                    continue

                texts.append(txt)

                metas.append({
                    "doc_id":     rec.get("doc_id"),
                    "chunk_id":   rec.get("chunk_id"),
                    "page_start": rec.get("page_start") or rec.get("pageStart")
                                  or (rec.get("meta") or {}).get("page_start"),
                    "page_end":   rec.get("page_end")   or rec.get("pageEnd")
                                  or (rec.get("meta") or {}).get("page_end"),
                })
    except FileNotFoundError:
        st.error(f"Missing chunks file: {chunks_path}")
        return [], []

    return texts, metas


def _ensure_embedder(state: Dict, model_name: str):
    """
    Return an Embedder instance, reusing state['embedder'] if present.
    Steps you implement:
      - If state.get('embedder') is None: create one with model_name and store it.
      - Return the embedder.
    """
    if state.get('embedder') is None:
      new_embedder = Embedder(model_name=model_name)
      state['embedder'] = new_embedder
      return new_embedder
    else:
        return state['embedder']

def _build_or_load_faiss(
    texts: List[str],
    metas: List[Dict],
    index_dir: Path,
    embedder, 
    rebuild: bool = True,
  ):
  """
  Build a FAISS index from texts (or load an existing one) and return (store, info_dict).
  Steps you implement:
    - Ensure index_dir exists.
    - If rebuild is False and an index file exists: load it via your FaissStore class and return.
    - Else:
        * Embed texts with your Embedder (batched).
        * Build FAISS (IndexFlatIP or your choice) via FaissStore.build(...)
        * Save index + meta into index_dir (e.g., input.index, input_meta.json)
    - Gather stats into a dict: {'ntotal': N, 'dim': D, 'path': str(index_path)}
    - Return (store, stats).
  """
  if not texts or not metas:
      raise ValueError("Texts or metas not defined properly.")
  index_dir.mkdir(parents=True, exist_ok=True)

  index_path = index_dir / "faiss.index"
  meta_path  = index_dir / "meta.json"

  if not rebuild and index_path.exists() and meta_path.exists():
    try:
        store = FaissStore(dim=0, index_path=index_path, meta_path=meta_path)
        store.load()
        store.dim = store.index.d
        stats = {
            "ntotal": store.index.ntotal,
            "dim": store.index.d,
            "index_path": str(index_path),
            "meta_path": str(meta_path),
        }
        return store, stats
    except Exception as e:
      print(f"Error: '{e}'. Falling back to rebuild... ") 
  
  embeddings = embedder.embed_texts(texts)
  assert len(embeddings) == len(texts) == len(metas)
  assert embeddings.ndim == 2
  if embeddings.dtype != np.float32:
      embeddings = embeddings.astype(np.float32)
  store = FaissStore(dim=embeddings.shape[1], index_path=index_path, meta_path=meta_path)
  store.build(embeddings=embeddings, metas=metas)
  assert store.index.ntotal == len(metas)
  stats = {
    "ntotal": store.index.ntotal,
    "dim": store.index.d,
    "index_path": str(index_path),
    "meta_path": str(meta_path),
    }   
  return store, stats

def _reset_index_state(state: Dict[str, object]) -> None:
    """
    Clear anything that ties Ask/Debug to a previous corpus.
    Call this right before committing a newly built index.
    """
    for key in ("store", "texts", "metas", "index_stats",
                "last_hits", "last_answer", "last_question", "used_context"):
        if key in state:
            try:
                del state[key]
            except Exception:
                state[key] = None  # belt & suspenders

def _make_corpus_id(selected_pages: List[Path], model_name: str) -> str:
    """
    Stable id for the selected files + embedding model.
    Uses abs paths + mtime + size so 'same set' maps to same folder.
    """
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    for p in sorted(map(Path, selected_pages)):
        try:
            s = p.stat()
            h.update(str(p.resolve()).encode("utf-8"))
            h.update(str(int(s.st_mtime)).encode("utf-8"))
            h.update(str(int(s.st_size)).encode("utf-8"))
        except FileNotFoundError:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:10]

def _index_stats_ui(stats: Dict) -> None:
  """
  Show a small stats block for the index.
  Renders:
    - metrics row: vectors (ntotal), dim, index filename
    - details expander: paths, existence, size, modified time
  """
  if not stats:
      st.info("No index loaded yet.")
      return

  ntotal = stats.get("ntotal")
  dim = stats.get("dim")
  index_path_str = stats.get("index_path")
  if not index_path_str:
      st.info("No index loaded yet.")
      return

  idx_p = Path(index_path_str)

  if ntotal == 0:
      st.warning("Index is empty—did you build after chunking?")

  if not idx_p.exists():
      st.error(f"Index file not found: {idx_p}")
      return

  meta_p = Path(stats.get("meta_path")) if stats.get("meta_path") else (idx_p.parent / "meta.json")
  index_name = idx_p.name

  # Top metrics row
  col1, col2, col3 = st.columns(3)
  with col1:
      st.metric("Vectors", f"{ntotal:,}" if isinstance(ntotal, int) and ntotal >= 0 else "—")
  with col2:
      st.metric("Dim", dim if isinstance(dim, int) and dim > 0 else "—")
  with col3:
      st.write(index_name)

  # Details table
  with st.expander("Details", expanded=False):
      idx_exists = idx_p.exists()
      meta_exists = meta_p.exists()

      def _size_mb(p: Path, exists: bool):
          return round(p.stat().st_size / (1024 * 1024), 2) if exists else "—"

      def _mtime_str(p: Path, exists: bool):
          return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else "—"

      rows = [
          {
              "File": "Index",
              "Path": str(idx_p),
              "Exists": idx_exists,
              "Size (MB)": _size_mb(idx_p, idx_exists),
              "Modified": _mtime_str(idx_p, idx_exists),
          },
          {
              "File": "Meta",
              "Path": str(meta_p),
              "Exists": meta_exists,
              "Size (MB)": _size_mb(meta_p, meta_exists),
              "Modified": _mtime_str(meta_p, meta_exists),
          },
      ]
      st.dataframe(rows, use_container_width=True)

      if not meta_exists:
          st.error("Meta file is missing.")

# ----- Implementing the UI -----
def render_chat_panel(
    state: AppState,
    k: int = 6,
    max_context_chars: int = 12000,   
    temperature: float = 0.20,         
) -> None:
    """
    Single-question Q&A panel with citations and evidence.
    Expects state['store'], state['texts'] set by the Index tab.
    Caches 'embedder' and 'llm' in state.
    """
    # --- Prechecks ---
    if state.get("store") is None or state.get("texts") is None:
        st.info("Build the index in the Index tab first.")
        return
    # sanity: texts and metas must align with the loaded store
    try:
        metas = getattr(state.get("store"), "metas", []) or []
        if len(state.get("texts") or []) != len(metas):
            st.warning("Index/text mismatch detected. Please rebuild the index.")
            return
    except Exception:
        pass

    # --- Input + trigger ---
    q = st.text_input("Ask a question about the indexed docs…", key="qa_key")
    submit = st.button("Answer", disabled=not q.strip())
    if not submit:
        return
    if not q.strip():
        st.info("Type a question…")
        return

    # --- Ensure embedder (match index dim / saved model name) ---
    store = state["store"]
    target_dim = getattr(getattr(store, "index", None), "d", None)
    embedder = state.get("embedder")

    need_new_embedder = (embedder is None) or (
        hasattr(embedder, "dim") and target_dim is not None and getattr(embedder, "dim", None) != target_dim
    )
    if need_new_embedder:
        # Prefer the model used during indexing, if the Index tab saved it.
        model_name = state.get("embed_model_name") or "BAAI/bge-small-en-v1.5"
        try:
            embedder = Embedder(model_name=model_name)
        except Exception as e:
            st.error(f"Failed to initialize embedder ({model_name}): {e}")
            return
        state["embedder"] = embedder

    # --- Ensure LLM client (cached) ---
    llm = state.get("llm")
    if llm is None:
        try:
            llm = OllamaClient(model="mistral:instruct")
        except Exception as e:
            st.error(f"Failed to initialize LLM client: {e}")
            return
        state["llm"] = llm

    # --- Retrieve + Generate ---
    with st.spinner("Retrieving + generating…"):
        try:
            out = answer_question(store, embedder, llm, q, state["texts"], k=k)
        except Exception as e:
            st.error(f"Operation failed: {e}")
            return

    # --- Render Answer ---
    st.subheader("Answer")
    st.markdown(out.get("answer", ""))

    citations = out.get("citations")
    if citations:
        st.caption("Citations: " + " ".join(citations))

    # --- Evidence table (top-k hits) ---
    def _page_tag(h: dict) -> str:
        ps = h.get("page_start") or h.get("pageStart") or (h.get("meta", {}) or {}).get("page_start")
        pe = h.get("page_end")   or h.get("pageEnd")   or (h.get("meta", {}) or {}).get("page_end")
        if isinstance(ps, int) and isinstance(pe, int):
            return f"[p{ps}–{pe}]" if pe != ps else f"[p{ps}]"
        if isinstance(ps, int):
            return f"[p{ps}]"
        return ""

    def _snippet(h: dict) -> str:
        txt = h.get("text")
        if not txt:
            # try pulling from original texts by index
            idx = h.get("idx") or h.get("i") or h.get("index")
            if isinstance(idx, int) and 0 <= idx < len(state["texts"]):
                txt = state["texts"][idx]
        if not isinstance(txt, str) or not txt:
            return ""
        snip = txt.strip().replace("\n", " ")
        return (snip[:160] + "…") if len(snip) > 160 else snip

    hits = out.get("used", []) or []
    rows = []
    for r, h in enumerate(hits[:k], start=1):
        score = h.get("score") or h.get("similarity") or h.get("distance")
        try:
            score_disp = f"{float(score):.3f}"
        except Exception:
            score_disp = "—"
        rows.append({
            "rank": r,
            "score": score_disp,
            "pages": _page_tag(h),
            "chunk_id": h.get("chunk_id") or (h.get("meta", {}) or {}).get("chunk_id"),
            "snippet": _snippet(h),
        })

    with st.expander("Evidence (top-k)"):
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.write("No evidence returned.")

    # --- Persist for Debug panel ---
    state["last_question"] = q
    state["last_answer"] = out.get("answer", "")
    state["last_hits"] = hits
    state["used_context"] = out.get("context")

def render_debug_panel( 
    state: AppState,
    top_n: int = 6,
    show_context_toggle_key: str = "show_context_blocks",
) -> None:
    """
    Renders a developer-facing table of retrieval and (optionally) the exact context.
    Expects state['last_hits'] (list of hit dicts) and optionally state['used_context'].
    """
    st.subheader("Debug · Retrieval & Context")

    hits = state.get("last_hits") or []
    if not hits:
        st.info("Ask a question in the Chat tab first to populate debug data.")
        return

    # --- helper formatters ---
    def _pages(h: dict) -> str:
        meta = h.get("meta") or {}
        ps = h.get("page_start", meta.get("page_start"))
        pe = h.get("page_end",   meta.get("page_end"))
        if isinstance(ps, int) and isinstance(pe, int):
            return f"[p{ps}–{pe}]" if pe != ps else f"[p{ps}]"
        if isinstance(ps, int):
            return f"[p{ps}]"
        return ""

    def _score(h: dict) -> str:
        s = h.get("score") or h.get("similarity") or h.get("distance")
        try:
            return f"{float(s):.3f}"
        except Exception:
            return "—"

    def _chunk_id(h: dict):
        return h.get("chunk_id") or (h.get("meta") or {}).get("chunk_id")

    def _first_line(h: dict) -> str:
        txt = h.get("text")
        if not isinstance(txt, str) or not txt:
            idx = h.get("idx") or h.get("index")
            texts = state.get("texts") or []
            if isinstance(idx, int) and 0 <= idx < len(texts):
                txt = texts[idx]
        if not isinstance(txt, str) or not txt:
            return ""
        oneline = txt.strip().splitlines()[0].strip()
        return (oneline[:160] + "…") if len(oneline) > 160 else oneline

    # --- table of top hits ---
    rows = []
    for r, h in enumerate(hits[:top_n], start=1):
        rows.append({
            "rank": r,
            "score": _score(h),
            "pages": _pages(h),
            "chunk_id": _chunk_id(h),
            "snippet": _first_line(h),
        })
    st.dataframe(rows, use_container_width=True)

    # Optional: show the exact context fed to the LLM (if captured)
    show_ctx = st.toggle("Show context blocks", key=show_context_toggle_key, value=False)
    if show_ctx:
        ctx = state.get("used_context")
        if isinstance(ctx, str) and ctx.strip():
            st.code(ctx, language="markdown")
        else:
            # Fallback: reconstruct a crude preview from hits
            blocks = []
            for h in hits[:top_n]:
                tag = _pages(h) or ""
                txt = h.get("text") or ""
                if not txt:
                    idx = h.get("idx") or h.get("index")
                    texts = state.get("texts") or []
                    if isinstance(idx, int) and 0 <= idx < len(texts):
                        txt = texts[idx]
                if isinstance(txt, str) and txt.strip():
                    blocks.append(f"{tag}\n{txt.strip()}")
            st.code("\n\n---\n\n".join(blocks) if blocks else "(no context captured)", language="markdown")

    # Question & answer echo (useful while iterating)
    with st.expander("Q/A echo"):
        st.markdown(f"**Q:** {state.get('last_question','')}")
        st.markdown(f"**A:** {state.get('last_answer','')}")

def main(
    interim_dir: Path = Path("data/interim"),
    index_dir: Path = Path("data/index"),
    default_model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    Orchestrates the 3-tab app: Upload, Index, Chat & Debug.
    """
    st.set_page_config(page_title="PDF Copilot", layout="wide")
    st.title("PDF Copilot")

    # Ensure shared state & create fresh session folders
    state = get_state()
    state.setdefault("paths", {})

    # One session-specific pair of folders (old session_* dirs wiped on first call)
    interim_dir_sess, index_dir_sess = _ensure_session_dirs(state, interim_dir, index_dir)

    # Remember session paths in state for other panels
    state["paths"]["interim_dir"] = interim_dir_sess
    state["paths"]["index_dir"]  = index_dir_sess

    # Default embed model; Index tab may overwrite after build
    state.setdefault("embed_model_name", default_model_name)

    tab_upload, tab_index, tab_chat, tab_debug = st.tabs(["Upload", "Index", "Ask", "Debug"])

    # --- Upload tab ---
    with tab_upload:
        # NOTE: pass the SESSION interim dir here
        render_upload_panel(state, interim_dir_sess)

    # --- Index tab ---
    with tab_index:
        st.subheader("Chunk & Index")
        # Controls for indexing session
        model_options = ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"]
        default_idx = model_options.index(state.get("embed_model_name", default_model_name)) \
                      if state.get("embed_model_name", default_model_name) in model_options else 0
        model_name = st.selectbox("Embedding model", model_options, index=default_idx, key="idx_model_name")
        rebuild = st.toggle("Rebuild index (ignore existing files)", value=True, key="idx_rebuild_toggle")

        # Run the index panel with chosen controls — pass SESSION dirs
        render_index_panel(
            state=state,
            interim_dir=interim_dir_sess,
            index_dir=index_dir_sess,
            model_name=model_name,
            rebuild=rebuild,
        )

        # If a store now exists, remember the model used for queries in Chat tab
        if state.get("store") is not None:
            state["embed_model_name"] = model_name

    # --- Chat tab ---
    with tab_chat:
        st.subheader("Ask")
        colA, colB = st.columns([1, 1])
        with colA:
            k_val = st.slider("Top-k", min_value=3, max_value=12, value=6, step=1,
                              help="How many chunks to retrieve")
        with colB:
            temp_val = st.slider("Temperature", min_value=0.0, max_value=0.9, value=0.0, step=0.1,
                                 help="Higher = more creative; lower = more precise")
        render_chat_panel(state, k=k_val, max_context_chars=12000, temperature=temp_val)

    # --- Debug tab ---
    with tab_debug:
        render_debug_panel(state, top_n=6)

        
if __name__ == "__main__":
    main()

# Running: 
# 
# streamlit run src\ui\app.py
