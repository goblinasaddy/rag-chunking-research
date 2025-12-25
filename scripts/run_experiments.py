import json
import os
from pathlib import Path
import csv
import time

from embeddings.embedder import EmbeddingGenerator
from retriever.dense_retriever import DenseRetriever
from rag.pipeline import RAGPipeline

from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.generation_metrics import compute_generation_metrics
from evaluation.hallucination import detect_hallucination


# =============================
# CONFIG
# =============================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable")


# =============================
# LOAD QUESTIONS
# =============================
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} questions.")


# =============================
# HELPER: count completed rows
# =============================
def count_completed_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # exclude header


# =============================
# MAIN EXPERIMENT LOOP
# =============================
for strategy in CHUNKING_STRATEGIES:
    print(f"\n=== Running experiments for: {strategy} ===")

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
        experiment_id=f"{strategy}_final"
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
    # RAG Pipeline
    # -------------------------
    rag = RAGPipeline(
        embedding_model_name=EMBEDDING_MODEL,
        retriever=retriever,
        gemini_api_key=GEMINI_API_KEY,
        llm_model_name=LLM_MODEL,
        temperature=0.0
    )

    # -------------------------
    # Output CSV (resumable)
    # -------------------------
    output_csv = RESULTS_DIR / f"{strategy}_results.csv"

    completed_rows = count_completed_rows(output_csv)
    print(f"Resuming from question {completed_rows + 1}")

    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "question",
            "gold_answer",
            "answer",
            "chunking_strategy",
            "recall_at_k",
            "precision_at_k",
            "hit_rate",
            "answer_correctness",
            "faithfulness",
            "hallucination",
            "hallucination_type",
            "latency_sec"
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

            start_time = time.time()
            result = rag.run(query)
            latency = time.time() - start_time

            retrieval_metrics = compute_retrieval_metrics(
                retrieved_chunks=result["retrieved_chunks"],
                gold_answer=gold_answer,
                k=TOP_K
            )

            generation_metrics = compute_generation_metrics(
                answer=result["answer"],
                gold_answer=gold_answer,
                context=result["context"]
            )

            hallucination_info = detect_hallucination(
                answer=result["answer"],
                context=result["context"],
                gold_answer=gold_answer
            )

            writer.writerow({
                "question": query,
                "gold_answer": gold_answer,
                "answer": result["answer"],
                "chunking_strategy": strategy,
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "precision_at_k": retrieval_metrics["precision_at_k"],
                "hit_rate": retrieval_metrics["hit_rate"],
                "answer_correctness": generation_metrics["answer_correctness"],
                "faithfulness": generation_metrics["faithfulness"],
                "hallucination": generation_metrics["hallucination"],
                "hallucination_type": hallucination_info["type"],
                "latency_sec": round(latency, 3)
            })

            csvfile.flush()  # CRITICAL: ensures progress is saved

            # Throttle to avoid 429s
            time.sleep(15)

    print(f"Saved results → {output_csv}")

print("\nAll experiments completed successfully.")
