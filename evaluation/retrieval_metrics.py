from typing import List, Dict


def is_relevant(chunk_text: str, gold_answer: str) -> bool:
    """
    Simple relevance check:
    A chunk is relevant if it contains the gold answer text (case-insensitive).
    """
    if not chunk_text or not gold_answer:
        return False

    return gold_answer.lower() in chunk_text.lower()


def recall_at_k(
    retrieved_chunks: List[Dict],
    gold_answer: str,
    k: int
) -> float:
    """
    Recall@k:
    Fraction of queries where at least one relevant chunk
    is retrieved in top-k.
    """
    top_k = retrieved_chunks[:k]

    for chunk in top_k:
        if is_relevant(chunk.get("text", ""), gold_answer):
            return 1.0

    return 0.0


def precision_at_k(
    retrieved_chunks: List[Dict],
    gold_answer: str,
    k: int
) -> float:
    """
    Precision@k:
    Number of relevant chunks in top-k divided by k.
    """
    top_k = retrieved_chunks[:k]
    if len(top_k) == 0:
        return 0.0

    relevant_count = sum(
        is_relevant(chunk.get("text", ""), gold_answer)
        for chunk in top_k
    )

    return relevant_count / k


def hit_rate(
    retrieved_chunks: List[Dict],
    gold_answer: str,
    k: int
) -> float:
    """
    Hit Rate@k:
    1 if any relevant chunk appears in top-k, else 0.
    (Equivalent to Recall@k for single-answer tasks.)
    """
    return recall_at_k(retrieved_chunks, gold_answer, k)


def compute_retrieval_metrics(
    retrieved_chunks: List[Dict],
    gold_answer: str,
    k: int
) -> Dict:
    """
    Compute all retrieval metrics for a single query.
    """
    return {
        "recall_at_k": recall_at_k(retrieved_chunks, gold_answer, k),
        "precision_at_k": precision_at_k(retrieved_chunks, gold_answer, k),
        "hit_rate": hit_rate(retrieved_chunks, gold_answer, k)
    }
