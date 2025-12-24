import json
from pathlib import Path
import os

import numpy as np

from embeddings.embedder import EmbeddingGenerator
from retriever.dense_retriever import DenseRetriever
from rag.pipeline import RAGPipeline


# -----------------------------
# CONFIG (TEMPORARY FOR TEST)
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKING_STRATEGY = "semantic"   # change to fixed / fixed_overlap / sentence
TOP_K = 5

# IMPORTANT: set your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable")


# -----------------------------
# LOAD CHUNKS
# -----------------------------
chunks_path = Path(f"data/processed/{CHUNKING_STRATEGY}_chunks.json")

if not chunks_path.exists():
    raise FileNotFoundError(
        f"Chunk file not found: {chunks_path}\n"
        f"Generate chunks before running this test."
    )

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

experiment_id = f"{CHUNKING_STRATEGY}_test"

embeddings, metadata = embedder.generate(
    chunks=chunks,
    experiment_id=experiment_id
)

print(f"Embeddings shape: {embeddings.shape}")


# -----------------------------
# RETRIEVER
# -----------------------------
embedding_dim = embeddings.shape[1]

retriever = DenseRetriever(
    embedding_dim=embedding_dim,
    top_k=TOP_K
)

retriever.build_index(embeddings, metadata)

print("FAISS index built.")


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
# TEST QUERY
# -----------------------------
query = "What is the minimum attendance requirement for students?"

result = rag.run(query)

print("\n==============================")
print("QUERY:")
print(result["query"])

print("\nRETRIEVED CHUNKS:")
for r in result["retrieved_chunks"]:
    print(
        f"- Rank {r['rank']} | Score {r['score']:.3f} | "
        f"Section {r['section_id']} | Tokens {r['token_count']}"
    )

print("\nANSWER:")
print(result["answer"])

print("\nLATENCY (sec):", round(result["latency_sec"], 2))
print("==============================")
