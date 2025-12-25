from typing import List, Dict
import time

import numpy as np
from sentence_transformers import SentenceTransformer

import google.generativeai as genai

from retriever.dense_retriever import DenseRetriever
from google.api_core.exceptions import ResourceExhausted



class RAGPipeline:
    """
    End-to-end RAG pipeline:
    query → retrieve → generate
    """

    def __init__(
        self,
        embedding_model_name: str,
        retriever: DenseRetriever,
        gemini_api_key: str,
        llm_model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0
    ):
        self.embedder = SentenceTransformer(
            embedding_model_name
        )
        self.retriever = retriever

        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel(llm_model_name)

        self.temperature = temperature

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedder.encode(
            query,
            normalize_embeddings=True
        )
        return embedding

    def build_context(self, retrieved_chunks: List[Dict]) -> str:
        context_blocks = []
        for r in retrieved_chunks:
            block = (
                f"[Chunk ID: {r['chunk_id']} | Section: {r['section_id']}]\n"
                f"{r.get('text', '')}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""
You are an AI assistant answering questions strictly
using the provided context.

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

            except ResourceExhausted as e:
                wait_time = 25
                print

    def run(self, query: str) -> Dict:
        """
        Execute full RAG pipeline for a single query.
        """
        start_time = time.time()

        query_embedding = self.embed_query(query)

        retrieved = self.retriever.retrieve(query_embedding)

        context = self.build_context(retrieved)

        answer = self.generate_answer(query, context)

        end_time = time.time()

        return {
            "query": query,
            "retrieved_chunks": retrieved,
            "context": context,
            "answer": answer,
            "latency_sec": end_time - start_time
        }
