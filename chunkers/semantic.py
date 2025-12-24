from typing import List, Dict
from pathlib import Path
import json

import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunker:
    """
    Semantic chunker based on sentence embedding similarity.

    Adjacent sentences are grouped together if their semantic
    similarity exceeds a threshold, subject to a token budget.
    """

    def __init__(
        self,
        embedding_model_name: str,
        tokenizer_name: str,
        similarity_threshold: float = 0.75,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

        self.embedder = SentenceTransformer(embedding_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )

    def _count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        return self.embedder.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def chunk_section(self, sentences: List[str]) -> List[str]:
        if len(sentences) == 0:
            return []

        embeddings = self._embed_sentences(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_tokens = self._count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sim = cosine_similarity(
                embeddings[i - 1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]

            sentence_tokens = self._count_tokens(sentences[i])

            if (
                sim >= self.similarity_threshold and
                current_tokens + sentence_tokens <= self.max_chunk_tokens
            ):
                current_chunk.append(sentences[i])
                current_tokens += sentence_tokens
            else:
                if current_tokens >= self.min_chunk_tokens:
                    chunks.append(" ".join(current_chunk))

                current_chunk = [sentences[i]]
                current_tokens = sentence_tokens

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
                    "chunking_strategy": "semantic",
                    "similarity_threshold": self.similarity_threshold
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

    chunker = SemanticChunker(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        max_chunk_tokens=512
    )

    chunks = chunker.chunk_document(processed_doc)
    print(f"Generated {len(chunks)} semantic chunks.")
