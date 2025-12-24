import json
import os
from pathlib import Path

from embeddings.embedder import EmbeddingGenerator
from retriever.dense_retriever import DenseRetriever
from rag.pipeline import RAGPipeline

from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.generation_metrics import compute_generation_metrics
from evaluation.hallucination import detect_hallucination


# -----------------------------
# CONFIG
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKING_STRATEGY = "semantic"  # try fixed / fixed_overlap / sentence / semantic
TOP_K = 5

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable")


# -----------------------------
# LOAD CHUNKS
# -----------------------------
chunks_path = Path(f"data/processed/{CHUNKING_STRATEGY}_chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks ({CHUNKING_STRATEGY})")


# -----------------------------
# EMBEDDINGS
# -----------------------------
embedder = EmbeddingGenerator(
    model_name=EMBEDDING_MODEL,
    device="cpu",
    normalize=True
)

embeddings, metadata = embedder.generate(
    chunks=chunks,
    experiment_id=f"{CHUNKING_STRATEGY}_eval"
)

retriever = DenseRetriever(
    embedding_dim=embeddings.shape[1],
    top_k=TOP_K
)
retriever.build_index(embeddings, metadata)


# -----------------------------
# RAG PIPELINE
# -----------------------------
rag = RAGPipeline(
    embedding_model_name=EMBEDDING_MODEL,
    retriever=retriever,
    gemini_api_key=GEMINI_API_KEY,
    llm_model_name="gemini-2.5-flash",
    temperature=0.0
)

# -----------------------------
# TEST QUERY + GOLD ANSWER
# -----------------------------
query = "What is the minimum attendance requirement for students?"
gold_answer = "75%"

result = rag.run(query)


# -----------------------------
# EVALUATION
# -----------------------------
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


# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n==============================")
print("QUERY:")
print(query)

print("\nANSWER:")
print(result["answer"])

print("\nRETRIEVAL METRICS:")
for k, v in retrieval_metrics.items():
    print(f"- {k}: {v}")

print("\nGENERATION METRICS:")
for k, v in generation_metrics.items():
    print(f"- {k}: {v}")

print("\nHALLUCINATION ANALYSIS:")
for k, v in hallucination_info.items():
    print(f"- {k}: {v}")

print("\nLATENCY (sec):", round(result["latency_sec"], 2))
print("==============================")
