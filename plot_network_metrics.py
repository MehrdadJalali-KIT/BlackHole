import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and filter data
df = pd.read_csv('results_aggregated.csv')
df = df[df['Method'].isin(['blackhole', 'stratified', 'pagerank', 'kcenter'])]
df['Accuracy_sem'] = df['Accuracy_Std'] / np.sqrt(1)
df['Cohen_Kappa_sem'] = df['Cohen_Kappa_Std'] / np.sqrt(1)

# Define visual styles
sns.set(style="whitegrid", context="notebook", palette="deep")
method_colors = {
    'blackhole': '#2E2E2E',
    'stratified': '#2CA02C',
    'pagerank': '#D62728',
    'kcenter': '#1F77B4'
}
method_styles = {
    'blackhole': '-',
    'stratified': '--',
    'pagerank': '-.',
    'kcenter': ':'
}
method_markers = {
    'blackhole': 'o',
    'stratified': '^',
    'pagerank': 's',
    'kcenter': 'D'
}

# Define metrics
performance_metrics = [('Accuracy_Mean', 'Accuracy_sem'), ('Cohen_Kappa_Mean', 'Cohen_Kappa_sem')]
network_metrics = [
    ('Modularity_Mean', 'Modularity_Std'),
    ('Num_Communities_Mean', 'Num_Communities_Std'),
    ('Avg_Community_Size_Mean', 'Avg_Community_Size_Std'),
    ('Avg_Clustering_Mean', 'Avg_Clustering_Std'),
    ('Graph_Density_Mean', 'Graph_Density_Std'),
    ('Avg_Degree_Mean', 'Avg_Degree_Std')
]
threshold_values = sorted(df['Threshold'].unique())
models = df['Model'].unique()

# Ensure output folder exists
output_dir = "plots_all_methods"
os.makedirs(output_dir, exist_ok=True)

# Clean plotting function (NO stars, NO % labels)
def plot_and_save_clean(subset, metric, std_metric, title_prefix):
    plt.figure(figsize=(9, 5))
    for method in ['blackhole', 'stratified', 'pagerank', 'kcenter']:
        method_data = subset[subset['Method'] == method].sort_values('Threshold')
        if method_data.empty:
            continue
        plt.errorbar(method_data['Threshold'], method_data[metric],
                     yerr=method_data[std_metric],
                     label=method.title(),
                     color=method_colors[method],
                     linestyle=method_styles[method],
                     marker=method_markers[method],
                     capsize=4, markersize=6)

    if metric in ['Accuracy_Mean', 'Cohen_Kappa_Mean']:
        strat_data = subset[subset['Method'] == 'stratified'].set_index('Threshold')
        if 0.0 in strat_data.index:
            baseline = strat_data.loc[0.0, metric]
            plt.axhline(y=baseline, color='gray', linestyle=':', label=f'Baseline (0.0) = {baseline:.3f}')

    metric_name = metric.replace("_Mean", "").replace("_", " ").title()
    plt.title(f'{title_prefix}: {metric_name}', fontsize=14)
    plt.xlabel('Pruning Rate', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(threshold_values)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=10)
    plt.tight_layout()

    filename = f"{output_dir}/{title_prefix}_{metric_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# Plot and save performance metrics
for model in models:
    subset = df[df['Model'] == model]
    for metric, std_metric in performance_metrics:
        plot_and_save_clean(subset, metric, std_metric, model)

# Plot and save network metrics (GAT only)
subset = df[df['Model'] == 'GAT']
for metric, std_metric in network_metrics:
    plot_and_save_clean(subset, metric, std_metric, 'Network')