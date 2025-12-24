from typing import List, Dict
from pathlib import Path
import json
import math

from transformers import AutoTokenizer


class FixedOverlapChunker:
    """
    Fixed-size token-based chunker with overlap.
    Overlap is implemented via sliding window stride.
    """

    def __init__(
        self,
        tokenizer_name: str,
        chunk_size: int,
        overlap_ratio: float = 0.2,
        min_chunk_tokens: int = 50
    ):
        assert 0.0 < overlap_ratio < 1.0, "overlap_ratio must be between 0 and 1"

        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_tokens = min_chunk_tokens

        self.stride = int(chunk_size * (1 - overlap_ratio))
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.chunk_size]

            if len(chunk_tokens) < self.min_chunk_tokens:
                break

            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            i += self.stride

        return chunks

    def chunk_document(self, processed_doc: List[Dict]) -> List[Dict]:
        chunked_output = []
        chunk_id = 0

        for section in processed_doc:
            text = section["text"]
            doc_id = section["doc_id"]
            section_id = section["section_id"]

            text_chunks = self.chunk_text(text)

            for chunk in text_chunks:
                token_count = len(
                    self.tokenizer.encode(chunk, add_special_tokens=False)
                )

                chunked_output.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "text": chunk,
                    "token_count": token_count,
                    "chunking_strategy": "fixed_overlap",
                    "overlap_ratio": self.overlap_ratio
                })

                chunk_id += 1

        return chunked_output


# -------------------------
# Standalone sanity check
# -------------------------
if __name__ == "__main__":
    processed_path = Path("data/processed/rulebook_processed.json")

    with open(processed_path, "r", encoding="utf-8") as f:
        processed_doc = json.load(f)

    chunker = FixedOverlapChunker(
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512,
        overlap_ratio=0.2
    )

    chunks = chunker.chunk_document(processed_doc)
    print(f"Generated {len(chunks)} fixed-overlap chunks.")
