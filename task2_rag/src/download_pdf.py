"""
Utility to download a public research paper (>= 30 pages) for Task 2 RAG benchmarking.
Defaults to 'Llama 2: Open Foundation and Fine-Tuned Chat Models' (arXiv:2307.09288, 77 pages).
"""

import os
import urllib.request
import fitz

DEFAULT_PDF_URL = "https://arxiv.org/pdf/2307.09288.pdf"
DEFAULT_SAVE_PATH = "./task2_rag/data/llama2_research_paper.pdf"

def download_benchmark_pdf(
    url: str = DEFAULT_PDF_URL,
    save_path: str = DEFAULT_SAVE_PATH,
    min_pages: int = 30
) -> str:
    """Download PDF and verify it has >= min_pages."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        doc = fitz.open(save_path)
        page_count = len(doc)
        doc.close()
        if page_count >= min_pages:
            print(f"Benchmark PDF already exists at '{save_path}' with {page_count} pages (>= {min_pages}).")
            return save_path
            
    print(f"Downloading public document collection from: {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(save_path, "wb") as out_file:
        out_file.write(response.read())
        
    doc = fitz.open(save_path)
    page_count = len(doc)
    doc.close()
    print(f"Downloaded '{save_path}' successfully. Verified page count: {page_count} pages.")
    
    if page_count < min_pages:
        raise ValueError(f"Downloaded PDF has {page_count} pages, which is less than the required {min_pages} pages.")
        
    return save_path

if __name__ == "__main__":
    download_benchmark_pdf()
