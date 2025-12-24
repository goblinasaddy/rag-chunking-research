import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates and caches embeddings for chunked documents.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
        cache_dir: str = "data/processed/embeddings"
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name, device=device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_paths(self, experiment_id: str) -> Tuple[Path, Path]:
        emb_path = self.cache_dir / f"{experiment_id}_embeddings.npy"
        meta_path = self.cache_dir / f"{experiment_id}_metadata.json"
        return emb_path, meta_path

    def generate(
        self,
        chunks: List[Dict],
        experiment_id: str,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate or load cached embeddings.

        Returns:
            embeddings: np.ndarray [num_chunks, embedding_dim]
            metadata: List[Dict] (aligned with embeddings)
        """
        emb_path, meta_path = self._cache_paths(experiment_id)

        if emb_path.exists() and meta_path.exists() and not force_recompute:
            embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return embeddings, metadata

        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize
        )

        metadata = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "section_id": c["section_id"],
                "token_count": c["token_count"],
                "chunking_strategy": c["chunking_strategy"],
                "text": c["text"]   # ← THIS MUST EXIST
            }
            for c in chunks
        ]

        np.save(emb_path, embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return embeddings, metadata


# -------------------------
# Standalone sanity check
# -------------------------
if __name__ == "__main__":
    # Example: embedding semantic chunks
    chunks_path = Path("data/processed/semantic_chunks.json")

    if not chunks_path.exists():
        print("No chunk file found for sanity check.")
        exit()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    embeddings, meta = embedder.generate(
        chunks=chunks,
        experiment_id="semantic_test"
    )

    print(f"Generated embeddings: {embeddings.shape}")
