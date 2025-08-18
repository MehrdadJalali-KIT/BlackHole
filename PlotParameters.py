# Ablation plots for gravity score weights (alpha, beta, gamma)
# - Reads: DiffentAlphaBetaGamma.csv
# - Produces: per-model line plots of Accuracy vs Threshold for selected weight triplets
# - Style: matplotlib only, grid + markers, no custom color names

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Load
# ----------------------
df = pd.read_csv("DiffentAlphaBetaGamma.csv")

# Keep Black Hole method rows (ablation applies to BH)
if "Method" in df.columns:
    df = df[df["Method"].str.lower() == "blackhole"].copy()

# Normalize/ensure columns
req = ["Threshold","Model","Alpha","Beta","Gamma","Accuracy"]
missing = [c for c in req if c not in df.columns]
assert not missing, f"CSV missing required columns: {missing}"

# Aggregate over runs if present
group_cols = ["Model","Alpha","Beta","Gamma","Threshold"]
agg = (df.groupby(group_cols, dropna=False)["Accuracy"]
         .agg(["mean","std","count"]).reset_index())
agg.rename(columns={"mean":"Accuracy_Mean","std":"Accuracy_Std","count":"N"}, inplace=True)

# ----------------------
# Helper: format triplet label
# ----------------------
def wlabel(a,b,g):
    return fr"$\alpha$={a:.2f}, $\beta$={b:.2f}, $\gamma$={g:.2f}"

# Choose canonical reference configs
canonical = [
    (1.00,0.00,0.00),  # degree only
    (0.00,1.00,0.00),  # modularity only
    (0.00,0.00,1.00),  # clustering only
    (0.33,0.33,0.34),  # balanced
]

# Ensure canonical weights exist in data (tolerate small float noise)
def exists_close(a,b,g):
    tol = 1e-3
    m = (np.isclose(agg["Alpha"], a, atol=tol) &
         np.isclose(agg["Beta"],  b, atol=tol) &
         np.isclose(agg["Gamma"], g, atol=tol))
    return m.any()

canon_in_data = [w for w in canonical if exists_close(*w)]

# Output dir
plots_dir = "plots_ablation"
os.makedirs(plots_dir, exist_ok=True)

models = sorted(agg["Model"].unique())

for model in models:
    sub = agg[agg["Model"]==model].copy()

    # Top-3 triplets by overall mean accuracy across thresholds
    overall = (sub.groupby(["Alpha","Beta","Gamma"])["Accuracy_Mean"]
                 .mean().reset_index().sort_values("Accuracy_Mean", ascending=False))
    top3 = [tuple(x) for x in overall.head(3)[["Alpha","Beta","Gamma"]].to_numpy()]

    # Selected weight sets = top3 + canonical (dedup, keep order)
    selected = []
    for w in top3 + canon_in_data:
        if w not in selected:
            selected.append(w)

    # --- Line plot: Accuracy vs. Threshold
    plt.figure(figsize=(10,6))
    for (a,b,g) in selected:
        mask = (np.isclose(sub["Alpha"],a,atol=1e-3) &
                np.isclose(sub["Beta"], b,atol=1e-3) &
                np.isclose(sub["Gamma"],g,atol=1e-3))
        line = sub[mask].sort_values("Threshold")
        if line.empty:
            continue
        plt.errorbar(
            line["Threshold"], line["Accuracy_Mean"],
            yerr=line["Accuracy_Std"],
            marker="o", linewidth=2, capsize=3, label=wlabel(a,b,g)
        )

    # Baseline: best Accuracy at Threshold=0.0
    base0 = sub[np.isclose(sub["Threshold"],0.0)]
    if not base0.empty:
        baseline = float(base0["Accuracy_Mean"].max())
        plt.axhline(baseline, linestyle=":", linewidth=1.5, label=f"Baseline (0.0) = {baseline:.3f}")

    plt.title(f"{model}: Accuracy vs. Pruning for Gravity-Score Weights", fontsize=18, fontweight="bold")
    plt.xlabel("Pruning Rate", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xticks(sorted(sub["Threshold"].unique()))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=11, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{model}_Ablation_Accuracy_vs_Threshold.png", dpi=600)
    plt.close()

    # --- Bar chart: overall ranking across thresholds (all weight triplets)
    rank = (sub.groupby(["Alpha","Beta","Gamma"])["Accuracy_Mean"]
              .mean().reset_index()
              .sort_values("Accuracy_Mean", ascending=False))
    rank["label"] = rank.apply(lambda r: f"{r.Alpha:.2f}/{r.Beta:.2f}/{r.Gamma:.2f}", axis=1)
    plt.figure(figsize=(10,6))
    plt.bar(rank["label"], rank["Accuracy_Mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Accuracy (across thresholds)")
    plt.title(f"{model}: Gravity-Score Weights — Overall Ranking", fontsize=18, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{model}_Ablation_Ranking.png", dpi=600)
    plt.close()

print(f"Saved ablation plots to: {plots_dir}")
