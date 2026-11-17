"""
PyMuPDF (fitz) text extraction module for PDF documents.
Ensures document page count verification (>= 30 pages) and structured page metadata.
"""

import os
from typing import List, Dict, Any
import fitz  # PyMuPDF

def extract_pdf_pages(pdf_path: str, min_pages: int = 30) -> List[Dict[str, Any]]:
    """
    Extract structured text from a PDF document using PyMuPDF.
    
    Args:
        pdf_path: Path to the target PDF file.
        min_pages: Minimum required pages to satisfy task constraints.
        
    Returns:
        List of dictionaries with page_number, text, and character count.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Loaded PDF '{os.path.basename(pdf_path)}' containing {total_pages} pages.")
    
    if total_pages < min_pages:
        raise ValueError(
            f"Document has {total_pages} pages, but at least {min_pages} pages are required."
        )
        
    extracted_pages = []
    total_chars = 0
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text")
        
        # Clean text: normalize newlines and whitespace
        lines = [line.strip() for line in text.split("\n")]
        cleaned_text = "\n".join([line for line in lines if line])
        
        extracted_pages.append({
            "page_number": page_idx + 1,  # 1-indexed for citations
            "text": cleaned_text,
            "char_count": len(cleaned_text)
        })
        total_chars += len(cleaned_text)
        
    doc.close()
    print(f"Extraction complete: {len(extracted_pages)} pages, {total_chars:,} total characters.")
    return extracted_pages
