
# 📘 Comparative Study of Chunking Strategies for Retrieval-Augmented Generation

This repository contains the full codebase, experiments, and analysis for the research paper:

> **A Comparative Study of Chunking Strategies for Retrieval-Augmented Generation in Policy-Oriented Documents**

The project presents a systematic, retrieval-focused evaluation of different document chunking strategies used in Retrieval-Augmented Generation (RAG) pipelines, with an emphasis on structured, fact-centric documents such as rulebooks and policy manuals.

---

## 🔍 Motivation

Retrieval-Augmented Generation (RAG) systems rely heavily on how source documents are segmented before indexing. While dense retrieval and embedding models have received extensive attention, **document chunking strategies are often treated as a heuristic preprocessing step**.

This project investigates a central research question:

> _How does chunking strategy influence retrieval performance in RAG systems when applied to policy-oriented, fact-centric documents?_

To answer this, we conduct controlled retrieval-only experiments that isolate the effect of chunking granularity from downstream language model generation.

---

## 🧠 Key Contributions

- Empirical comparison of **four chunking strategies** under a unified retrieval framework
    
- Controlled **retrieval-only evaluation** to isolate chunking effects
    
- Analysis on a **policy-oriented university rulebook**
    
- Evaluation using standard IR metrics: Recall@k, Precision@k, Hit Rate, and Latency
    
- Insight that **sentence-based chunking outperforms semantic chunking** for fact-centric retrieval tasks
    

---

## 🧩 Chunking Strategies Evaluated

1. **Fixed Chunking**  
    Segments text into fixed-length chunks without overlap.
    
2. **Fixed Chunking with Overlap**  
    Adds overlap between adjacent chunks to reduce boundary effects.
    
3. **Sentence-Based Chunking**  
    Splits the document at sentence boundaries to preserve atomic factual units.
    
4. **Semantic Chunking**  
    Groups sentences into variable-length chunks based on embedding similarity.
    

---

## 🏗️ Project Structure

rag-chunking-research/
│
├── data/
│   ├── raw/
│   │   └── rulebook.pdf              # Source document
│   └── processed/
│       ├── fixed_chunks.json
│       ├── fixed_overlap_chunks.json
│       ├── sentence_chunks.json
│       ├── semantic_chunks.json
│       └── questions.json            # Evaluation queries + gold answers
│
├── chunkers/
│   ├── fixed.py
│   ├── fixed_overlap.py
│   ├── sentence.py
│   └── semantic.py
│
├── embeddings/
│   └── embedder.py                   # Sentence-transformer embedding logic
│
├── retriever/
│   └── dense_retriever.py            # FAISS-based dense retrieval
│
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── generation_metrics.py
│   └── hallucination.py
│
├── rag/
│   └── pipeline.py                   # Full RAG pipeline (generation optional)
│
├── scripts/
│   ├── test_pipeline.py
│   └── run_experiments.py            # Retrieval-only experiment runner
│
├── analysis/
│   ├── aggregate_retrieval_results.py
│   └── plot_retrieval_results.py
│
├── results/
│   ├── tables/
│   │   ├── fixed_retrieval_only.csv
│   │   ├── fixed_overlap_retrieval_only.csv
│   │   ├── sentence_retrieval_only.csv
│   │   └── semantic_retrieval_only.csv
│   └── plots/
│       ├── mean_recall_at_k.png
│       ├── mean_precision_at_k.png
│       └── mean_latency_sec.png
│
├── README.md
├── requirements.txt
└── references.bib


---

## ⚙️ Experimental Setup

- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence-Transformers)
    
- **Retriever:** FAISS (cosine similarity)
    
- **Evaluation Mode:** Retrieval-only (no LLM generation)
    
- **Queries:** 67 fact-centric questions derived from the document
    
- **Metrics:** Recall@k, Precision@k, Hit Rate, Retrieval Latency
    

All chunking strategies are evaluated under identical conditions to ensure fair comparison.

---

## 📊 Key Results (Summary)

| Chunking Strategy | Mean Recall@k | Mean Precision@k |
| ----------------- | ------------- | ---------------- |
| Sentence          | **0.58**      | **0.14**         |
| Fixed             | 0.42          | 0.10             |
| Fixed + Overlap   | 0.39          | 0.12             |
| Semantic          | 0.34          | 0.10             |


**Observation:**  
Sentence-based chunking consistently outperforms more complex semantic segmentation strategies for structured, fact-centric documents.

---

## 📈 Visualizations

The repository includes publication-ready plots comparing retrieval performance across chunking strategies:

- Mean Recall@k
    
- Mean Precision@k
    
- Retrieval Latency
    

Plots are available under `results/plots/` and are used directly in the research paper.

---

## 📄 Paper

The full paper is written in LaTeX (IEEE conference format) and includes:

- Introduction & Related Work
    
- Methodology & Experimental Setup
    
- Results & Discussion
    
- Limitations & Future Work
    

The paper is intended for submission to **arXiv** and relevant **NLP / IR workshops**.

---

## 🧪 Reproducibility

To reproduce experiments:

1. Place the source PDF in `data/raw/`
    
2. Generate chunks using the scripts in `chunkers/`
    
3. Run retrieval-only experiments:
    
    `python -m scripts.run_experiments`
    
4. Aggregate and visualize results:
    
    `python -m analysis.aggregate_retrieval_results python -m analysis.plot_retrieval_results`
    

---

## 👤 Author

**Aditya Kumar Singh**  
B.Tech Artificial Intelligence  
Amity University, Greater Noida  
Email: jr.aditya004@gmail.com

---

## 📜 License

This project is released for academic and research use.  
Please cite the accompanying paper if you use this work.

---

## ⭐ Acknowledgements

This work was conducted as an independent research project.  
The author thanks the open-source NLP and IR community for tools and libraries that made this study possible.