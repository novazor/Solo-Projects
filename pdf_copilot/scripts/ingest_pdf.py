# # test script for pdf ingestion
from src.ingest.pdf_loader import *
from pathlib import Path

def main():
    """
    Steps:
      1) Read input path
      2) call load_pdf()
      3) run clean_page_text() on each page
      4) write to data/interim/<doc_name>_pages.jsonl
      5) print: total pages, sample of first 400 chars from page 1
    """
    pdf_path = Path(r"S:\Code\rfx_copilot\data\raw\input.pdf")
    doc = load_pdf(pdf_path)

    for page in doc:
        page["text"] = clean_page_text(page["text"])
        
    out = Path("data/interim") / f"{pdf_path.stem}_pages.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(doc, out_path=out)

    print(f"Num pages: {len(doc)}")
    print(doc[0]['text'][:200])
    
if __name__ == "__main__":
    main()

