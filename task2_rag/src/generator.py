"""
Answer generation module with explicit source citations [Page X, Chunk Y].
"""

from typing import List, Dict, Any, Optional

RAG_PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful and precise technical assistant. Answer the user question based strictly on the provided context passages.
Every claim in your answer must cite its source passage in the format [Page X, Chunk Y].
If the context does not contain sufficient information to answer the question, state that clearly.
<|im_end|>
<|im_start|>user
Context Passages:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

def build_context_string(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into indexed context blocks."""
    blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "unknown")
        pnum = c.get("page_number", "?")
        text = c.get("text", "").strip()
        blocks.append(f"[{cid} | Page {pnum}]:\n{text}")
    return "\n\n".join(blocks)

class RAGGenerator:
    """Generates answers with grounded source citations."""
    def __init__(self, model=None, tokenizer=None, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Generate answer with source citations from retrieved chunks.
        """
        context_str = build_context_string(retrieved_chunks)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)
        
        citations = [
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "excerpt": c["text"][:100] + "..."
            }
            for c in retrieved_chunks
        ]
        
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            gen_tokens = outputs[0][input_len:]
            answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            # Fallback extractive summary when generator model is not loaded
            top_chunk = retrieved_chunks[0] if retrieved_chunks else None
            if top_chunk:
                p = top_chunk.get("page_number")
                cid = top_chunk.get("chunk_id")
                answer = (
                    f"Based on the retrieved context from [Page {p}, {cid}], "
                    f"the relevant information is: \"{top_chunk['text'][:200]}...\""
                )
            else:
                answer = "No relevant context found in source documents."
                
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved_count": len(retrieved_chunks)
        }
