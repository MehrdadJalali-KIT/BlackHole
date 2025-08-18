# ==========================================
# Redundancy Impact Plots (mean only, no shadows)
# - Reads: Redundency_aggregated_results.csv
# - Plots: Accuracy vs. Redundancy Level (per model)
# - Aggregation: Mean across runs
# - Style schema matches your previous code
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Config
# ------------------------------
INPUT_CSV  = "Redundency_aggregated_results.csv"
OUTPUT_DIR = "plots_redundancy_mean"

sns.set(style="whitegrid", context="talk", palette="deep")

METHOD_ORDER  = ["blackhole", "stratified", "pagerank", "kcenter"]
METHOD_COLORS = {
    "blackhole": "#2E2E2E",
    "stratified": "#2CA02C",
    "pagerank":   "#D62728",
    "kcenter":    "#1F77B4",
}
METHOD_STYLES = {
    "blackhole": "-",
    "stratified": "--",
    "pagerank":   "-.",
    "kcenter":    ":",
}
METHOD_MARKERS = {
    "blackhole": "o",
    "stratified": "^",
    "pagerank":   "s",
    "kcenter":    "D",
}

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv(INPUT_CSV)

# Required columns
needed = ["Redundancy_Level", "Method", "Model", "Accuracy_Mean"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

# Normalize method names
df["Method"] = df["Method"].str.lower()

# ------------------------------
# Aggregate: Mean per (Model, Method, Redundancy_Level)
# ------------------------------
agg = (
    df.groupby(["Model", "Method", "Redundancy_Level"], dropna=False)
      .agg(Accuracy_Mean=("Accuracy_Mean", "mean"))
      .reset_index()
)

# ------------------------------
# Plot function (mean only)
# ------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_accuracy_vs_redundancy(model_name, subset):
    plt.figure(figsize=(10, 6))
    methods_here = [m for m in METHOD_ORDER if m in subset["Method"].unique()]

    for method in methods_here:
        md = subset[subset["Method"] == method].sort_values("Redundancy_Level")
        if md.empty:
            continue

        # Line: mean only
        plt.plot(
            md["Redundancy_Level"], md["Accuracy_Mean"],
            label=method.title(),
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_STYLES.get(method, "-"),
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2, markersize=8
        )

    # Baseline: Stratified at redundancy 0 (mean)
    base = subset[(subset["Method"] == "stratified") & (np.isclose(subset["Redundancy_Level"], 0.0))]
    if not base.empty:
        baseline = float(base["Accuracy_Mean"].iloc[0])
        plt.axhline(
            baseline, color="gray", linestyle=":", linewidth=1.5,
            label=f"Baseline (Stratified @ 0.0) = {baseline:.3f}"
        )

    plt.title(f"{model_name}: Accuracy vs. Redundancy Level", fontsize=20, fontweight="bold")
    plt.xlabel("Redundancy Level", fontsize=18)
    plt.ylabel("Accuracy (Mean)", fontsize=18)
    plt.xticks(sorted(subset["Redundancy_Level"].unique()), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=14)
    plt.tight_layout()

    fname = f"{OUTPUT_DIR}/{model_name}_Accuracy_vs_Redundancy.png"
    plt.savefig(fname, dpi=600)
    plt.close()

# ------------------------------
# Generate plots per model
# ------------------------------
for model in sorted(agg["Model"].unique()):
    plot_accuracy_vs_redundancy(model, agg[agg["Model"] == model])

print(f"Saved redundancy mean-only plots to: {OUTPUT_DIR}")
