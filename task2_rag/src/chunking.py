"""
Token-based chunking module supporting dual chunk sizes and sliding-window token overlap.
"""

from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

DEFAULT_TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class TokenChunker:
    """Tokenizes text and produces sliding-window token-bounded text chunks."""
    
    def __init__(self, tokenizer_name: str = DEFAULT_TOKENIZER_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def chunk_document(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Split a sequence of pages into token-bounded chunks.
        
        Args:
            pages: List of dicts with 'page_number' and 'text'.
            chunk_size: Maximum token count per chunk.
            overlap: Number of tokens overlapping between adjacent chunks.
            
        Returns:
            List of chunk records with unique IDs and source metadata.
        """
        assert chunk_size > overlap, "chunk_size must be strictly greater than overlap"
        stride = chunk_size - overlap
        
        chunks = []
        chunk_counter = 0
        
        for p in pages:
            page_num = p["page_number"]
            page_text = p["text"].strip()
            if not page_text:
                continue
                
            # Tokenize page text
            encoding = self.tokenizer(
                page_text,
                add_special_tokens=False,
                return_offsets_mapping=True
            )
            input_ids = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]
            total_tokens = len(input_ids)
            
            if total_tokens == 0:
                continue
                
            # Slice with sliding window
            for start_idx in range(0, total_tokens, stride):
                end_idx = min(start_idx + chunk_size, total_tokens)
                chunk_token_ids = input_ids[start_idx:end_idx]
                
                # Decode exact chunk text
                chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True).strip()
                if len(chunk_text) < 20:  # Skip trivial fragments
                    continue
                    
                chunk_record = {
                    "chunk_id": f"chunk_{chunk_size}_{chunk_counter:04d}",
                    "page_number": page_num,
                    "text": chunk_text,
                    "token_count": len(chunk_token_ids),
                    "chunk_size_setting": chunk_size,
                    "overlap_setting": overlap,
                    "start_token_idx": start_idx,
                    "end_token_idx": end_idx
                }
                chunks.append(chunk_record)
                chunk_counter += 1
                
                if end_idx == total_tokens:
                    break
                    
        print(f"Generated {len(chunks)} chunks for chunk_size={chunk_size} (overlap={overlap}).")
        return chunks
