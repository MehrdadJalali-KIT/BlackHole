import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Config
# ================================
INPUT_CSV = "sparsification_performance.csv"
OUTPUT_DIR = "plots_efficiency_thresholds"

# 4 method styling (same schema/format as before)
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

# ================================
# Load & normalize schema
# ================================
df = pd.read_csv(INPUT_CSV)

# Ensure required key columns
if "Threshold" not in df.columns or "Method" not in df.columns:
    raise ValueError("CSV must contain 'Threshold' and 'Method' columns.")

# If Model is missing or NaN for parts (e.g., sparsification phase), fill for grouping
if "Model" not in df.columns:
    df["Model"] = "GLOBAL"
else:
    df["Model"] = df["Model"].fillna("GLOBAL")

# Normalize metric column names across old/new schemas
# We will plot these three performance parameters:
#   1) Graph construction time (sparsification cost)
#   2) Training time
#   3) Peak memory
metric_map = {}

# Graph construction time
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

# Validate we found all three
required_keys = ["Graph_Construct_s", "Train_Time_s", "Peak_Memory_MB"]
missing = [k for k in required_keys if k not in metric_map]
if missing:
    raise ValueError(
        f"Could not find required metrics {missing} in CSV. "
        f"Columns available: {list(df.columns)}"
    )

# Keep only relevant columns
keep_cols = ["Model", "Method", "Threshold", "Run"]
for _, (colname, _) in metric_map.items():
    keep_cols.append(colname)
keep_cols = list(dict.fromkeys(keep_cols))  # de-dup
df = df[keep_cols]

# ================================
# Aggregate mean ± std per (Model, Method, Threshold)
# ================================
agg_dict = {metric_map["Graph_Construct_s"][0]: ["mean", "std"],
            metric_map["Train_Time_s"][0]: ["mean", "std"],
            metric_map["Peak_Memory_MB"][0]: ["mean", "std"]}

agg = (
    df.groupby(["Model", "Method", "Threshold"], dropna=False)
      .agg(agg_dict)
      .reset_index()
)

# Flatten MultiIndex columns
agg.columns = ["Model", "Method", "Threshold"] + [
    f"{col[0]}_{col[1].capitalize()}" for col in agg.columns[3:]
]

# If we have "GLOBAL" rows (e.g., sparsification-only records), duplicate to each actual model
models_present = sorted([m for m in agg["Model"].unique() if m != "GLOBAL"])
if "GLOBAL" in agg["Model"].unique() and models_present:
    global_rows = agg[agg["Model"] == "GLOBAL"].copy()
    dup_rows = []
    for m in models_present:
        tmp = global_rows.copy()
        tmp["Model"] = m
        dup_rows.append(tmp)
    if dup_rows:
        agg = pd.concat([agg[agg["Model"] != "GLOBAL"]] + dup_rows, ignore_index=True)

# Final list of models to plot
models_to_plot = sorted(agg["Model"].unique())

# ================================
# Plot function (same schema/format)
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

def titlecase_metric_label(raw_label):
    # Already friendly names passed in, just return
    return raw_label

def plot_metric_vs_threshold(subset, value_col, std_col, metric_label, model_name):
    plt.figure(figsize=(10, 6))
    methods_here = [m for m in METHOD_ORDER if m in subset["Method"].unique()]
    # Draw lines with error bars
    for method in methods_here:
        md = subset[subset["Method"] == method].sort_values("Threshold")
        if md.empty:
            continue
        plt.errorbar(
            md["Threshold"], md[value_col],
            yerr=md[std_col],
            label=method.title(),
            color=METHOD_COLORS.get(method, None),
            linestyle=METHOD_STYLES.get(method, "-"),
            marker=METHOD_MARKERS.get(method, "o"),
            capsize=4, markersize=8, linewidth=2
        )

    # Baseline line for stratified @ 0.0 if present
    base = subset[(subset["Method"] == "stratified") & (np.isclose(subset["Threshold"], 0.0))]
    if not base.empty:
        baseline = float(base[value_col].iloc[0])
        plt.axhline(baseline, color="gray", linestyle=":", linewidth=1.5,
                    label=f"Baseline (stratified @ 0.0) = {baseline:.3f}")

    plt.title(f"{model_name}: {titlecase_metric_label(metric_label)}", fontsize=20, fontweight="bold")
    plt.xlabel("Pruning Rate", fontsize=18)
    plt.ylabel(metric_label, fontsize=18)
    xticks = sorted(subset["Threshold"].unique().tolist())
    plt.xticks(xticks, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=14)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{model_name}_{metric_label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fname, dpi=600)
    plt.close()

# ================================
# Generate plots (3 metrics × models)
# ================================
metrics_to_plot = [
    ("Graph_Construct_s", metric_map["Graph_Construct_s"][0], "Graph Construction Time (s)"),
    ("Train_Time_s",      metric_map["Train_Time_s"][0],      "Training Time (s)"),
    ("Peak_Memory_MB",    metric_map["Peak_Memory_MB"][0],    "Peak Memory (MB)"),
]

for model in models_to_plot:
    model_df = agg[agg["Model"] == model]
    for key, raw_col, nice_label in metrics_to_plot:
        mean_col = f"{raw_col}_Mean"
        std_col  = f"{raw_col}_Std"
        if mean_col not in model_df.columns or std_col not in model_df.columns:
            # Skip if metric not present in this schema
            continue
        plot_metric_vs_threshold(model_df[["Method","Threshold",mean_col,std_col]].copy(),
                                 mean_col, std_col, nice_label, model)

print(f"Saved plots to: {OUTPUT_DIR}")
