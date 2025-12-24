from typing import Dict


def normalize(text: str) -> str:
    """
    Normalize text for comparison.
    """
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def answer_correctness(answer: str, gold_answer: str) -> float:
    """
    Answer correctness:
    1.0 if the model answer contains the gold answer
    (or vice versa), else 0.0
    """
    answer_n = normalize(answer)
    gold_n = normalize(gold_answer)

    if not answer_n or not gold_n:
        return 0.0

    if gold_n in answer_n or answer_n in gold_n:
        return 1.0

    return 0.0


def faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness:
    1.0 if the answer is fully supported by retrieved context.
    """
    answer_n = normalize(answer)
    context_n = normalize(context)

    if not answer_n or not context_n:
        return 0.0

    # Check that key answer fragments appear in context
    answer_tokens = answer_n.split()
    supported_tokens = sum(
        1 for token in answer_tokens if token in context_n
    )

    support_ratio = supported_tokens / max(len(answer_tokens), 1)

    # Conservative threshold
    return 1.0 if support_ratio >= 0.6 else 0.0


def hallucination(answer: str, context: str) -> float:
    """
    Hallucination:
    1.0 if hallucination detected, else 0.0
    """
    # Hallucination = NOT faithful
    return 1.0 - faithfulness(answer, context)


def compute_generation_metrics(
    answer: str,
    gold_answer: str,
    context: str
) -> Dict:
    """
    Compute all generation-level metrics for a single query.
    """
    correctness = answer_correctness(answer, gold_answer)
    faithful = faithfulness(answer, context)

    return {
        "answer_correctness": correctness,
        "faithfulness": faithful,
        "hallucination": 1.0 - faithful
    }
