"""
pdf_loader.py

Read a PDF from disk and return page records ready for chunking.
Uses PyMuPDF (fitz) for fast, layout-aware text extraction.

Exports:
- load_pdf(path, doc_id=None, password=None) -> list of {doc_id, page_num, text}
- save_pages_jsonl(pages, out_path) -> out_path
- guess_doc_id(path) -> str

Details:
- page_num is 1-based.
- Trims whitespace and normalizes newlines; leaves images/tables alone.
- Encrypted PDFs: supply password via arg; raises on failure.
"""

from pathlib import Path
from typing import List, Dict
import fitz
import json
import re

def load_pdf(path: Path) -> List[Dict]:
    """
    Returns a list of pages from a PDF.
    Each item: {"doc_id": str, "page_num": int, "text": str}
    TODO:
      - Use a fast PDF lib (prob PyMuPDF or something like that)
      - Extract text per page
      - Keep original page order
      - return the final list of pages
    """
    page_list = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1): 
            text = page.get_text()
            page_list.append({
                "doc_id": path.stem,  
                "page_num": i,
                "text": text  
            })
    return page_list  

def clean_page_text(text: str) -> str:
    """
    Cleans page text to rid of unnecessary fluff.
    Returns a single string of the cleaned text
    TODO:
      - Strip of unnecessary whitespace/newlines, clear symbols
      - Use regex for matching
      - return cleaned string
    """
    cleaned_string = text.replace("\r\n", "\n")
    cleaned_string = cleaned_string.replace("\t", "    ")
    cleaned_string = re.sub(r'(\n\s*)+\n', '\n\n', cleaned_string)

    lines = cleaned_string.splitlines()
    trimmed_lines = [line.rstrip() for line in lines]
    cleaned_string = "\n".join(trimmed_lines)
    if text.endswith('\n') and not cleaned_string.endswith('\n'):
        cleaned_string += '\n'
    return cleaned_string

def save_jsonl(pages: List[Dict], out_path: Path) -> None:
    """
    Saves a jsonl file for all pages in the input dictionary
    Writes to the inputted path - returns nothing.
    TODO:
      - Loop through the pages and write json line by line
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in pages:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")