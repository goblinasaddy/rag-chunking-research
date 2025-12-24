from typing import Dict, List


def normalize(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def extract_tokens(text: str) -> List[str]:
    return normalize(text).split()


def detect_hallucination(
    answer: str,
    context: str,
    gold_answer: str = ""
) -> Dict:
    """
    Detect and categorize hallucination in a model answer.
    Returns a structured explanation.
    """

    answer_n = normalize(answer)
    context_n = normalize(context)
    gold_n = normalize(gold_answer)

    if not answer_n:
        return {
            "hallucination": True,
            "type": "empty_answer",
            "reason": "Model produced no answer."
        }

    # Token overlap with context
    answer_tokens = extract_tokens(answer_n)
    context_tokens = set(extract_tokens(context_n))

    supported_tokens = [
        tok for tok in answer_tokens if tok in context_tokens
    ]

    support_ratio = len(supported_tokens) / max(len(answer_tokens), 1)

    # Case 1: Fully grounded
    if support_ratio >= 0.6:
        return {
            "hallucination": False,
            "type": "no_hallucination",
            "reason": "Answer is well supported by retrieved context."
        }

    # Case 2: Mentions gold answer but adds extras
    if gold_n and gold_n in answer_n and support_ratio >= 0.3:
        return {
            "hallucination": True,
            "type": "unsupported_detail",
            "reason": (
                "Answer contains the correct core information "
                "but introduces additional details not supported by context."
            )
        }

    # Case 3: Very low grounding
    if support_ratio < 0.3:
        return {
            "hallucination": True,
            "type": "fabricated_specifics",
            "reason": (
                "Answer contains specific claims that are not supported "
                "by retrieved context."
            )
        }

    # Fallback
    return {
        "hallucination": True,
        "type": "overgeneralization",
        "reason": (
            "Answer is loosely related to context but overgeneralizes "
            "beyond available evidence."
        )
    }
