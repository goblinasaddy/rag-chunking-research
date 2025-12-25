import pandas as pd
from pathlib import Path


# =============================
# CONFIG
# =============================
RESULTS_DIR = Path("results/tables")
OUTPUT_PATH = Path("results/aggregated_retrieval_results.csv")

FILES = {
    "fixed": "fixed_retrieval_only.csv",
    "fixed_overlap": "fixed_overlap_retrieval_only.csv",
    "sentence": "sentence_retrieval_only.csv",
    "semantic": "semantic_retrieval_only.csv",
}


# =============================
# AGGREGATION
# =============================
rows = []

for strategy, filename in FILES.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    rows.append({
        "chunking_strategy": strategy,
        "mean_recall_at_k": df["recall_at_k"].mean(),
        "std_recall_at_k": df["recall_at_k"].std(),
        "mean_precision_at_k": df["precision_at_k"].mean(),
        "std_precision_at_k": df["precision_at_k"].std(),
        "mean_hit_rate": df["hit_rate"].mean(),
        "std_hit_rate": df["hit_rate"].std(),
        "mean_latency_sec": df["retrieval_latency_sec"].mean(),
    })


summary_df = pd.DataFrame(rows)

# Sort by recall (primary metric)
summary_df = summary_df.sort_values(
    by="mean_recall_at_k",
    ascending=False
)

# Save
summary_df.to_csv(OUTPUT_PATH, index=False)

print("\n=== Aggregated Retrieval Results ===")
print(summary_df.round(4))
print(f"\nSaved aggregated results → {OUTPUT_PATH}")
