# PDF Copilot — RAG for PDF Doc Answers with Citations

## One-liner

A local, privacy-friendly assistant that ingests pdf documents, answers questions with **page-level citations**, and constructs structured responses—powered by a \~7–8B local LLM.

---

## Outcomes

* **Business value:** Faster RFP reading, fewer missed requirements/deadlines.
* **Engineering value (resume-ready):** Table-aware ingestion, hybrid retrieval (BM25 + dense + rerank), faithfulness/no-answer guard, evaluation harness (RAGAS + IR metrics), deployable demo.

**Success criteria**

* Q\&A returns the correct clause **with page(s) cited** ≥80% on a 30–50 Q/A gold set.
* “No answer” triggers correctly for out-of-scope questions.
* Structured extraction returns valid JSON with page references.

---

## Scope

**In scope**

* PDF text extraction (+ OCR fallback later)
* Paragraph/sentence chunking with overlap; table extraction to markdown/CSV
* Hybrid retrieval: FAISS dense + BM25 (+ optional cross-ref expansion)
* CPU reranker for top-k precision
* Local LLM answering (Mistral-7B-Instruct or Llama-3-8B) with citations
* Streamlit UI, Docker packaging, small evaluation harness

**Out of scope (v1)**

* Multi-tenant auth, RBAC
* Automatic PDF redaction
* Fine-tuning models

---

## Users & Primary Tasks

* **Proposal writers/analysts**: “What’s the submission deadline? Cite the clause.”
* **Project managers**: “List mandatory deliverables + pages.”
* **Contracts/procurement**: “Is a performance bond required? Show the section.”

**Demo queries**

* “Where do I submit the proposal?”
* “What is the protest deadline and how is it calculated?”
* “Summarize evaluation criteria with weights as a table.”
* “Make a JSON checklist of mandatory forms with page numbers.”

---

## Data

* Public RFP PDFs (city/state/federal) + appendices (tables, forms)
* Optional FAR excerpts for cross-document grounding
* A curated 30–50 Q/A gold set per doc (question, expected answer, page)

---

## Architecture (high level)

```
PDFs → Ingestion → Pages JSONL
             ↓
       Chunker (overlap, table-aware)
             ↓
   Embeddings (bge-small) → FAISS
             ↑                  ↓
     BM25 (Whoosh/rank_bm25)   Query
                   \          /
                    Fusion + Rerank (bge-reranker-base CPU)
                              ↓
                      Context Pack
                              ↓
  Local LLM (Mistral-7B/Llama-3-8B via Ollama/llama.cpp)
                              ↓
            Answer + Page Citations + (optional) JSON Extract
```

---

## Components

1. **Ingestion**

   * PyMuPDF for text pages; (later) OCR fallback (Tesseract/PaddleOCR)
   * Camelot/Tabula for tables → markdown/CSV chunks
   * Metadata: doc\_id, page\_num, table\_id, section header (best-effort)

2. **Chunking**

   * Paragraph-first, target \~1.0–1.3k chars, 10–15% overlap
   * De-hyphenation & headings retained
   * Page span per chunk

3. **Retrieval**

   * Dense: `BAAI/bge-small-en-v1.5` (cosine; normalized; FAISS IndexFlatIP)
   * BM25: Whoosh or `rank_bm25` (lightweight)
   * Fusion: normalized sum or Reciprocal Rank Fusion
   * Rerank: `bge-reranker-base` on top-20 → keep top-k (6–8)

4. **Answering**

   * Local LLM (\~7–8B) via **Ollama**/**llama.cpp** (GGUF Q4\_K\_M/Q5\_0)
   * Prompt enforces: “Answer only from context; cite pages; say ‘I don’t know’ otherwise.”
   * Faithfulness verifier (mini self-ask or rule: low max sim ⇒ no-answer)

5. **Structured extraction**

   * Pydantic schema:

     ```json
     {
       "due_dates": [{"text": "...", "page": 3}],
       "deliverables": [...],
       "mandatory_requirements": [...],
       "evaluation_criteria": [{"criterion":"...","weight":"...","page":6}]
     }
     ```
   * Retry on invalid JSON; drop items without page

6. **Evaluation**

   * RAGAS: faithfulness, answer relevancy
   * IR: Recall\@k, MRR
   * Scripted report artifact (CSV/Markdown)

7. **UI**

   * Streamlit: upload docs, chat box, toggle Q\&A/Extract
   * Show top-k chunks and clickable citations (page preview/snippet)

---

## Tech Stack (8GB-friendly)

* **LLM:** Mistral-7B-Instruct or Llama-3-8B (Ollama/llama.cpp, Q4\_K\_M)
* **Embeddings:** `bge-small-en-v1.5`
* **Reranker:** `bge-reranker-base` (CPU)
* **Vector store:** FAISS
* **BM25:** rank\_bm25 or Whoosh
* **PDF:** PyMuPDF (+ Tesseract/PaddleOCR later), Camelot/Tabula
* **UI:** Streamlit
* **Eval:** RAGAS
* **Packaging:** Docker + Makefile (stretch)

---

## Folder Structure

```
rfx_copilot/
  data/
    raw/         # PDFs
    interim/     # pages.jsonl, chunks.jsonl, tables
    index/       # faiss index + meta
  src/
    ingest/      # pdf_loader.py, ocr.py (later)
    chunk/       # text_chunker.py
    embed/       # embedder.py
    index_store/ # faiss_store.py
    retrieve/    # fusion, rerank, query api (later)
    generate/    # llm_client.py, prompts.py (later)
    ui/          # streamlit app (later)
    eval/        # ragas_runner.py
  scripts/
    ingest_pdf.py
    make_chunks.py
    build_index.py
    search_chunks.py
    eval_run.py
  tests/
  README.md
  proj_plan.md
  requirements.txt
```

---

## Environment & Performance Notes

* Use CPU for embeddings & reranker; keep GPU/VRAM for the LLM.
* Generation params: `max_new_tokens=256–512`, `temperature=0.2–0.5`.
* If memory tight, run LLM on Q4\_K\_M and keep FAISS + reranker on CPU.

---

## Milestones & Definition of Done

### Milestone 1 — Vertical Slice (Done when…)

* [ ] `pages.jsonl` produced
* [ ] `chunks.jsonl` produced (avg \~1–1.5k chars; overlap verified)
* [ ] FAISS index built; `search_chunks.py` returns sensible top-k

### Milestone 2 — Retrieval Quality

* [ ] BM25 integrated; fusion improves Recall\@20 vs dense-only
* [ ] CPU reranker improves top-k precision
* [ ] “No-answer” rule implemented

### Milestone 3 — LLM Answering + Citations

* [ ] Local LLM answers with page-level citations
* [ ] At least one case correctly returns “I don’t know”

### Milestone 4 — Structured Extraction

* [ ] Valid JSON per Pydantic schema with pages
* [ ] 5–10 accurate items per category on sample RFP

### Milestone 5 — UI & Evaluation

* [ ] Streamlit app with upload, chat, toggle Q\&A/Extract, citations
* [ ] RAGAS + IR metrics report over 30–50 questions
* [ ] README with architecture diagram + demo GIF

---

## Evaluation Plan

* Build a gold set (30–50 Q/A) from one RFP.
* Metrics:

  * **IR:** Recall\@k (k=5/10/20), nDCG\@k
  * **Gen:** RAGAS faithfulness & answer relevancy
  * **UX:** Latency p50/p95 (retrieval + generation)
* Report template:

  * Baseline (dense only) → +BM25 → +rerank → +no-answer
  * Show deltas per step

---

## Risks & Mitigations

* **Scanned PDFs** → OCR fallback; mark low-confidence pages
* **Tables lost in text** → Camelot to separate table chunks; query-hint routing to include at least one table
* **Hallucinations** → strict prompt + no-answer threshold + verifier pass
* **Windows path/import pain** → run CLIs with `python -m ...` from project root

---

## Stretch Goals

* Cross-reference graph (e.g., “See Section 3.2”) → neighborhood expansion
* Query routing: Answer vs Extract vs Lookup-far-section
* Long-context model A/B with retrieval
* Redis caching for hot queries

---

## Demo Script (5 minutes)

1. Upload sample RFP.
2. Ask: “What is the submission deadline?” → shows date/time + **page**.
3. Ask: “Is a performance bond required?” → “Not required” + **Section 7**.
4. Switch to **Extract** → produce JSON with `due_dates`, `deliverables`, etc., each with pages.
5. Show “no-answer” on “What color is the mayor’s car?”

---

## Resume Bullets (pick 2–3)

* Built a **local procurement RAG copilot** that answers complex RFP questions with **audited page citations**, combining **BM25 + dense retrieval + CPU reranker** powered by a **7–8B quantized LLM** on an 8GB GPU.
* Designed a **table-aware ingestion pipeline** (PyMuPDF + Camelot) with hierarchical chunking and cross-ref expansion, improving **Recall\@20 by \~X%** and **RAGAS faithfulness by \~Y%** over a naive baseline.
* Shipped a **Streamlit + Docker** app with **structured extraction** (deadlines/deliverables JSON) and a reproducible **evaluation harness** (RAGAS + IR metrics).

---

## Timeline (2–3 weeks part-time)

* **Week 1:** Milestone 1
* **Week 2:** Milestones 2–3
* **Week 3:** Milestones 4–5 + polish


