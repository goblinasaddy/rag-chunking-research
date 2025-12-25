import json
import csv
import time
from pathlib import Path

from embeddings.embedder import EmbeddingGenerator
from retriever.dense_retriever import DenseRetriever
from evaluation.retrieval_metrics import compute_retrieval_metrics


# =============================
# CONFIG
# =============================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

CHUNKING_STRATEGIES = [
    "fixed",
    "fixed_overlap",
    "sentence",
    "semantic"
]

QUESTIONS_PATH = Path("data/processed/questions.json")
CHUNKS_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# LOAD QUESTIONS
# =============================
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} questions.")


# =============================
# HELPER
# =============================
def count_completed_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header


# =============================
# MAIN LOOP
# =============================
for strategy in CHUNKING_STRATEGIES:
    print(f"\n=== Retrieval-only experiments for: {strategy} ===")

    chunks_path = CHUNKS_DIR / f"{strategy}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunk file: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # -------------------------
    # Embeddings
    # -------------------------
    embedder = EmbeddingGenerator(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        normalize=True
    )

    embeddings, metadata = embedder.generate(
        chunks=chunks,
        experiment_id=f"{strategy}_retrieval_only"
    )

    # -------------------------
    # Retriever
    # -------------------------
    retriever = DenseRetriever(
        embedding_dim=embeddings.shape[1],
        top_k=TOP_K
    )
    retriever.build_index(embeddings, metadata)

    # -------------------------
    # CSV (resumable)
    # -------------------------
    output_csv = RESULTS_DIR / f"{strategy}_retrieval_only.csv"
    completed_rows = count_completed_rows(output_csv)
    print(f"Resuming from question {completed_rows + 1}")

    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "question",
            "gold_answer",
            "chunking_strategy",
            "recall_at_k",
            "precision_at_k",
            "hit_rate",
            "retrieval_latency_sec"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # -------------------------
        # Question loop
        # -------------------------
        for idx, qa in enumerate(questions):
            if idx < completed_rows:
                continue

            print(f"[{strategy}] Question {idx + 1}/{len(questions)}")

            query = qa["question"]
            gold_answer = qa["answer"]

            start = time.time()

            # 🔹 Explicit query embedding
            query_embedding = embedder.embed_query(query)
            retrieved_chunks = retriever.retrieve(query_embedding)

            latency = time.time() - start

            metrics = compute_retrieval_metrics(
                retrieved_chunks=retrieved_chunks,
                gold_answer=gold_answer,
                k=TOP_K
            )

            writer.writerow({
                "question": query,
                "gold_answer": gold_answer,
                "chunking_strategy": strategy,
                "recall_at_k": metrics["recall_at_k"],
                "precision_at_k": metrics["precision_at_k"],
                "hit_rate": metrics["hit_rate"],
                "retrieval_latency_sec": round(latency, 4)
            })

            csvfile.flush()

    print(f"Saved results → {output_csv}")

print("\nAll retrieval-only experiments completed successfully.")
