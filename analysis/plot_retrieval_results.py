import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =============================
# CONFIG
# =============================
INPUT_PATH = Path("results/aggregated_retrieval_results.csv")
OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# LOAD DATA
# =============================
df = pd.read_csv(INPUT_PATH)

# Ensure consistent ordering
df = df.sort_values("mean_recall_at_k", ascending=False)


# =============================
# PLOT 1: Mean Recall@k
# =============================
plt.figure(figsize=(8, 5))
plt.bar(
    df["chunking_strategy"],
    df["mean_recall_at_k"]
)
plt.ylabel("Mean Recall@k")
plt.xlabel("Chunking Strategy")
plt.title("Retrieval Recall@k by Chunking Strategy")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_recall_at_k.png", dpi=300)
plt.show()


# =============================
# PLOT 2: Mean Precision@k
# =============================
plt.figure(figsize=(8, 5))
plt.bar(
    df["chunking_strategy"],
    df["mean_precision_at_k"]
)
plt.ylabel("Mean Precision@k")
plt.xlabel("Chunking Strategy")
plt.title("Retrieval Precision@k by Chunking Strategy")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_precision_at_k.png", dpi=300)
plt.show()


# =============================
# PLOT 3: Mean Retrieval Latency
# =============================
plt.figure(figsize=(8, 5))
plt.bar(
    df["chunking_strategy"],
    df["mean_latency_sec"]
)
plt.ylabel("Mean Retrieval Latency (seconds)")
plt.xlabel("Chunking Strategy")
plt.title("Retrieval Latency by Chunking Strategy")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_latency_sec.png", dpi=300)
plt.show()


print("\nPlots saved to:", OUTPUT_DIR)
