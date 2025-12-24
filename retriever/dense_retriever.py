from typing import List, Dict, Tuple
import numpy as np
import faiss


class DenseRetriever:
    """
    Dense vector retriever using FAISS.
    Assumes embeddings are L2-normalized for cosine similarity.
    """

    def __init__(self, embedding_dim: int, top_k: int = 5):
        self.embedding_dim = embedding_dim
        self.top_k = top_k

        # Inner product index (cosine similarity when vectors are normalized)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Build FAISS index from embeddings and aligned metadata.
        """
        assert embeddings.shape[0] == len(metadata), \
            "Embeddings and metadata size mismatch"

        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

    def retrieve(
        self,
        query_embedding: np.ndarray
    ) -> List[Dict]:
        """
        Retrieve top-k most similar chunks for a single query embedding.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            self.top_k
        )

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            result = {
                "rank": rank + 1,
                "score": float(scores[0][rank]),
                "chunk_id": self.metadata[idx]["chunk_id"],
                "doc_id": self.metadata[idx]["doc_id"],
                "section_id": self.metadata[idx]["section_id"],
                "token_count": self.metadata[idx]["token_count"],
                "chunking_strategy": self.metadata[idx]["chunking_strategy"],
                "text": self.metadata[idx].get("text", "")
            }

            results.append(result)

        return results


# -------------------------
# Standalone sanity check
# -------------------------
if __name__ == "__main__":
    # Dummy sanity check
    dim = 384
    retriever = DenseRetriever(embedding_dim=dim, top_k=3)

    # Fake embeddings (already normalized)
    embeddings = np.random.rand(10, dim).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    metadata = [
        {
            "chunk_id": i,
            "doc_id": "rulebook",
            "section_id": i,
            "token_count": 100,
            "chunking_strategy": "fixed"
        }
        for i in range(10)
    ]

    retriever.build_index(embeddings, metadata)

    query = np.random.rand(dim)
    query /= np.linalg.norm(query)

    results = retriever.retrieve(query)
    print("Retrieved results:")
    for r in results:
        print(r)
