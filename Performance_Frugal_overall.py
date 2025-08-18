import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Config
# ================================
INPUT_CSV = "sparsification_performance.csv"
OUTPUT_DIR = "plots_efficiency_overall"

sns.set(style="whitegrid", context="talk", palette="deep")
METHOD_ORDER  = ["blackhole", "stratified", "pagerank", "kcenter"]
METHOD_COLORS = {
    "blackhole": "#2E2E2E",
    "stratified": "#2CA02C",
    "pagerank":   "#D62728",
    "kcenter":    "#1F77B4",
}

# ================================
# Load & normalize schema
# ================================
df = pd.read_csv(INPUT_CSV)

# Ensure essentials
if "Method" not in df.columns or "Threshold" not in df.columns:
    raise ValueError("CSV must contain 'Method' and 'Threshold' columns.")

if "Model" not in df.columns:
    df["Model"] = "GLOBAL"
else:
    df["Model"] = df["Model"].fillna("GLOBAL")

# Detect metric columns (new and legacy)
metric_map = {}

# Graph construction time (sparsification cost)
if "Graph_Construct_s" in df.columns:
    metric_map["Graph_Construct_s"] = ("Graph_Construct_s", "Graph Construction Time (s)")
elif "Sparsification_Time" in df.columns:
    metric_map["Graph_Construct_s"] = ("Sparsification_Time", "Graph Construction Time (s)")

# Training time
if "Train_Time_s" in df.columns:
    metric_map["Train_Time_s"] = ("Train_Time_s", "Training Time (s)")
elif "Training_Time" in df.columns:
    metric_map["Train_Time_s"] = ("Training_Time", "Training Time (s)")

# Peak memory (prefer overall → training → legacy)
if "Overall_Peak_Memory_MB" in df.columns:
    metric_map["Peak_Memory_MB"] = ("Overall_Peak_Memory_MB", "Peak Memory (MB)")
elif "Peak_Memory_Train_MB" in df.columns:
    metric_map["Peak_Memory_MB"] = ("Peak_Memory_Train_MB", "Peak Memory (MB)")
elif "Peak_Memory_MB" in df.columns:
    metric_map["Peak_Memory_MB"] = ("Peak_Memory_MB", "Peak Memory (MB)")

required = ["Graph_Construct_s", "Train_Time_s", "Peak_Memory_MB"]
missing = [k for k in required if k not in metric_map]
if missing:
    raise ValueError(f"Missing required metrics {missing}. Found columns: {list(df.columns)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convenience
def method_ordered(df_in):
    # Keep only known methods, preserve defined order
    methods = [m for m in METHOD_ORDER if m in df_in["Method"].unique()]
    return df_in.set_index("Method").loc[methods].reset_index()

# Baseline for dotted line (stratified @ 0.0), computed from raw rows
def compute_baseline_mean(metric_col):
    base = df[(df["Method"] == "stratified") & (np.isclose(df["Threshold"], 0.0))]
    if metric_col in base.columns and not base[metric_col].dropna().empty:
        return float(base[metric_col].mean())
    return None

# ================================
# 1) Overall bars across all models & thresholds
# ================================
def plot_overall_bar(metric_key):
    raw_col, nice_label = metric_map[metric_key]
    use = df[["Method", raw_col]].dropna()
    if use.empty:
        return
    agg = (use.groupby("Method")[raw_col]
           .agg(["mean", "std"])
           .reset_index())
    agg = method_ordered(agg)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        agg["Method"].str.title(),
        agg["mean"],
        yerr=agg["std"],
        capsize=5,
        color=[METHOD_COLORS.get(m, "#555555") for m in agg["Method"]],
        alpha=0.9
    )

    # Annotate bars
    for b, v in zip(bars, agg["mean"]):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                 f"{v:.2f}", ha="center", va="bottom", fontsize=12)

    # Baseline line (stratified@0.0) if available
    base_val = compute_baseline_mean(raw_col)
    if base_val is not None:
        plt.axhline(base_val, color="gray", linestyle=":", linewidth=1.5,
                    label=f"Baseline (stratified@0.0) = {base_val:.2f}")

    plt.title(f"Overall Comparison: {nice_label}", fontsize=20, fontweight="bold")
    plt.ylabel(nice_label, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if base_val is not None:
        plt.legend(fontsize=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(OUTPUT_DIR, f"Overall_{nice_label.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.savefig(fname, dpi=600)
    plt.close()

# ================================
# 2) Per-model overall bars (one figure per metric)
# ================================
def plot_overall_bar_per_model(metric_key):
    raw_col, nice_label = metric_map[metric_key]
    models = sorted([m for m in df["Model"].unique() if m != "GLOBAL"])
    if not models:  # no distinct models logged
        return

    for model in models:
        use = df[df["Model"] == model][["Method", raw_col]].dropna()
        if use.empty:
            continue
        agg = (use.groupby("Method")[raw_col]
               .agg(["mean", "std"])
               .reset_index())
        agg = method_ordered(agg)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            agg["Method"].str.title(),
            agg["mean"],
            yerr=agg["std"],
            capsize=5,
            color=[METHOD_COLORS.get(m, "#555555") for m in agg["Method"]],
            alpha=0.9
        )
        for b, v in zip(bars, agg["mean"]):
            plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=12)

        # Model-specific baseline
        base = df[(df["Model"] == model) & (df["Method"] == "stratified") & (np.isclose(df["Threshold"], 0.0))]
        if raw_col in base.columns and not base[raw_col].dropna().empty:
            base_val = float(base[raw_col].mean())
            plt.axhline(base_val, color="gray", linestyle=":", linewidth=1.5,
                        label=f"Baseline (stratified@0.0) = {base_val:.2f}")
        else:
            base_val = None

        plt.title(f"{model}: Overall {nice_label}", fontsize=20, fontweight="bold")
        plt.ylabel(nice_label, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if base_val is not None:
            plt.legend(fontsize=12)
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        fname = os.path.join(OUTPUT_DIR, f"{model}_Overall_{nice_label.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plt.savefig(fname, dpi=600)
        plt.close()

# ================================
# 3) One 3-panel overall figure (all metrics together)
# ================================
def plot_overall_three_panel():
    # Build aggregates for each metric
    panels = []
    for metric_key in ["Graph_Construct_s", "Train_Time_s", "Peak_Memory_MB"]:
        raw_col, nice_label = metric_map[metric_key]
        use = df[["Method", raw_col]].dropna()
        if use.empty:
            panels.append(None)
            continue
        agg = (use.groupby("Method")[raw_col]
               .agg(["mean", "std"])
               .reset_index())
        agg = method_ordered(agg)
        panels.append((agg, nice_label))

    # If everything is empty, bail
    if all(p is None for p in panels):
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = []
    for i, pack in enumerate(panels):
        ax = axes[i]
        if pack is None:
            ax.axis("off")
            continue
        agg, nice_label = pack
        bars = ax.bar(
            agg["Method"].str.title(),
            agg["mean"],
            yerr=agg["std"],
            capsize=5,
            color=[METHOD_COLORS.get(m, "#555555") for m in agg["Method"]],
            alpha=0.9
        )
        for b, v in zip(bars, agg["mean"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11)
        # Global baseline line
        base_val = compute_baseline_mean(metric_map[[ "Graph_Construct_s", "Train_Time_s", "Peak_Memory_MB" ][i]][0])
        if base_val is not None:
            ax.axhline(base_val, color="gray", linestyle=":", linewidth=1.5)
        ax.set_title(nice_label, fontsize=16, fontweight="bold")
        ax.set_ylabel(nice_label, fontsize=14)
        ax.set_xticklabels(agg["Method"].str.title(), fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        titles.append(nice_label)

    fig.suptitle("Overall Comparison (All Models & Thresholds)", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(OUTPUT_DIR, "Overall_Three_Metrics.png")
    plt.savefig(fname, dpi=600)
    plt.close()

# ================================
# Run all overall plots
# ================================
for k in ["Graph_Construct_s", "Train_Time_s", "Peak_Memory_MB"]:
    plot_overall_bar(k)
    plot_overall_bar_per_model(k)

plot_overall_three_panel()

print(f"Saved overall plots to: {OUTPUT_DIR}")
