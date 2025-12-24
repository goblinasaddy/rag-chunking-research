from typing import List, Dict
from pathlib import Path
import json

from transformers import AutoTokenizer


class FixedChunker:
    """
    Fixed-size token-based chunker.

    This chunker splits text into contiguous token blocks
    of a fixed maximum size with NO overlap.
    """

    def __init__(
        self,
        tokenizer_name: str,
        chunk_size: int,
        min_chunk_tokens: int = 50
    ):
        self.chunk_size = chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Tokenize text and split into fixed-size chunks.
        """
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]

            if len(chunk_tokens) < self.min_chunk_tokens:
                continue

            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def chunk_document(self, processed_doc: List[Dict]) -> List[Dict]:
        """
        Apply fixed-size chunking to a processed document.
        """
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
                    "chunking_strategy": "fixed"
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

    chunker = FixedChunker(
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512
    )

    chunks = chunker.chunk_document(processed_doc)
    print(f"Generated {len(chunks)} fixed-size chunks.")
