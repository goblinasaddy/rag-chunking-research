import time
from typing import List, Dict

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from embeddings.embedder import EmbeddingGenerator
from retriever.dense_retriever import DenseRetriever


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    Query -> Retrieval -> Context assembly -> LLM generation

    IMPORTANT:
    - This file contains NO top-level execution.
    - Safe to import in retrieval-only experiments.
    """

    def __init__(
        self,
        embedding_model_name: str,
        retriever: DenseRetriever,
        gemini_api_key: str,
        llm_model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        device: str = "cpu",
    ):
        self.embedder = EmbeddingGenerator(
            model_name=embedding_model_name,
            device=device,
            normalize=True,
        )

        self.retriever = retriever
        self.temperature = temperature

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel(llm_model_name)

    # -------------------------------------------------
    # Retrieval
    # -------------------------------------------------
    def retrieve(self, query: str) -> List[Dict]:
        """
        Embed query and retrieve top-k chunks.
        """
        query_embedding = self.embedder.embed_query(query)
        return self.retriever.retrieve(query_embedding)

    # -------------------------------------------------
    # Context construction
    # -------------------------------------------------
    @staticmethod
    def build_context(retrieved_chunks: List[Dict]) -> str:
        """
        Concatenate retrieved chunks into a single context string.
        """
        return "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

    # -------------------------------------------------
    # Generation (Gemini)
    # -------------------------------------------------
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using Gemini with retry on quota exhaustion.
        """
        prompt = f"""
You are an AI assistant answering questions strictly using the provided context.

If the answer is not present in the context,
reply with: "The information is not available in the document."

Context:
{context}

Question:
{query}

Answer:
"""

        while True:
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature
                    }
                )
                return response.text.strip()

            except ResourceExhausted:
                wait_time = 30
                print(f"[RATE LIMIT] Gemini quota hit. Sleeping {wait_time}s...")
                time.sleep(wait_time)

    # -------------------------------------------------
    # Full pipeline
    # -------------------------------------------------
    def run(self, query: str) -> Dict:
        """
        Run full RAG pipeline for a single query.
        """
        start_time = time.time()

        retrieved_chunks = self.retrieve(query)
        context = self.build_context(retrieved_chunks)
        answer = self.generate_answer(query, context)

        latency = time.time() - start_time

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved_chunks,
            "latency_sec": latency,
        }














# import json
# import csv
# import time
# from pathlib import Path

# from embeddings.embedder import EmbeddingGenerator
# from retriever.dense_retriever import DenseRetriever
# from evaluation.retrieval_metrics import compute_retrieval_metrics


# # =============================
# # CONFIG
# # =============================
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# TOP_K = 5

# CHUNKING_STRATEGIES = [
#     "fixed",
#     "fixed_overlap",
#     "sentence",
#     "semantic"
# ]

# QUESTIONS_PATH = Path("data/processed/questions.json")
# CHUNKS_DIR = Path("data/processed")
# RESULTS_DIR = Path("results/tables")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# # =============================
# # LOAD QUESTIONS
# # =============================
# with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
#     questions = json.load(f)

# print(f"Loaded {len(questions)} questions.")


# # =============================
# # HELPER: count completed rows
# # =============================
# def count_completed_rows(csv_path: Path) -> int:
#     if not csv_path.exists():
#         return 0
#     with open(csv_path, "r", encoding="utf-8") as f:
#         return max(0, sum(1 for _ in f) - 1)  # exclude header


# # =============================
# # MAIN LOOP
# # =============================
# for strategy in CHUNKING_STRATEGIES:
#     print(f"\n=== Retrieval-only experiments for: {strategy} ===")

#     chunks_path = CHUNKS_DIR / f"{strategy}_chunks.json"
#     if not chunks_path.exists():
#         raise FileNotFoundError(f"Missing chunk file: {chunks_path}")

#     with open(chunks_path, "r", encoding="utf-8") as f:
#         chunks = json.load(f)

#     # -------------------------
#     # Embeddings
#     # -------------------------
#     embedder = EmbeddingGenerator(
#         model_name=EMBEDDING_MODEL,
#         device="cpu",
#         normalize=True
#     )

#     embeddings, metadata = embedder.generate(
#         chunks=chunks,
#         experiment_id=f"{strategy}_retrieval_only"
#     )

#     # -------------------------
#     # Retriever
#     # -------------------------
#     retriever = DenseRetriever(
#         embedding_dim=embeddings.shape[1],
#         top_k=TOP_K
#     )
#     retriever.build_index(embeddings, metadata)

#     # -------------------------
#     # Output CSV (resumable)
#     # -------------------------
#     output_csv = RESULTS_DIR / f"{strategy}_retrieval_only.csv"
#     completed_rows = count_completed_rows(output_csv)
#     print(f"Resuming from question {completed_rows + 1}")

#     write_header = not output_csv.exists()

#     with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
#         fieldnames = [
#             "question",
#             "gold_answer",
#             "chunking_strategy",
#             "recall_at_k",
#             "precision_at_k",
#             "hit_rate",
#             "retrieval_latency_sec"
#         ]

#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if write_header:
#             writer.writeheader()

#         # -------------------------
#         # Question loop
#         # -------------------------
#         for idx, qa in enumerate(questions):
#             if idx < completed_rows:
#                 continue

#             print(f"[{strategy}] Question {idx + 1}/{len(questions)}")

#             query = qa["question"]
#             gold_answer = qa["answer"]

#             start = time.time()
#             retrieved_chunks = retriever.retrieve(query)
#             latency = time.time() - start

#             retrieval_metrics = compute_retrieval_metrics(
#                 retrieved_chunks=retrieved_chunks,
#                 gold_answer=gold_answer,
#                 k=TOP_K
#             )

#             writer.writerow({
#                 "question": query,
#                 "gold_answer": gold_answer,
#                 "chunking_strategy": strategy,
#                 "recall_at_k": retrieval_metrics["recall_at_k"],
#                 "precision_at_k": retrieval_metrics["precision_at_k"],
#                 "hit_rate": retrieval_metrics["hit_rate"],
#                 "retrieval_latency_sec": round(latency, 4)
#             })

#             csvfile.flush()

#     print(f"Saved results → {output_csv}")

# print("\nAll retrieval-only experiments completed successfully.")
