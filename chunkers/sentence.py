from typing import List, Dict
from pathlib import Path
import json

from transformers import AutoTokenizer


class SentenceChunker:
    """
    Sentence-based chunker.

    Groups full sentences together until a maximum token
    budget is reached. Sentences are never split.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )

    def _count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

    def chunk_section(self, sentences: List[str]) -> List[str]:
        """
        Build chunks by accumulating sentences up to max_chunk_tokens.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If a single sentence exceeds max tokens, skip it
            if sentence_tokens > self.max_chunk_tokens:
                continue

            if current_tokens + sentence_tokens <= self.max_chunk_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # finalize current chunk
                if current_tokens >= self.min_chunk_tokens:
                    chunks.append(" ".join(current_chunk))

                # start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        # Add final chunk
        if current_tokens >= self.min_chunk_tokens:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_document(self, processed_doc: List[Dict]) -> List[Dict]:
        chunked_output = []
        chunk_id = 0

        for section in processed_doc:
            doc_id = section["doc_id"]
            section_id = section["section_id"]
            sentences = section["sentences"]

            section_chunks = self.chunk_section(sentences)

            for chunk_text in section_chunks:
                token_count = self._count_tokens(chunk_text)

                chunked_output.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "text": chunk_text,
                    "token_count": token_count,
                    "chunking_strategy": "sentence"
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

    chunker = SentenceChunker(
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_tokens=512
    )

    chunks = chunker.chunk_document(processed_doc)
    print(f"Generated {len(chunks)} sentence-based chunks.")
