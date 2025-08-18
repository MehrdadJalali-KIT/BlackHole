import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools

# Load and filter data
df = pd.read_csv('results_aggregated.csv')
df = df[df['Method'].isin(['blackhole', 'stratified'])]
df['Accuracy_sem'] = df['Accuracy_Std'] / np.sqrt(1)
df['Cohen_Kappa_sem'] = df['Cohen_Kappa_Std'] / np.sqrt(1)

# Define visual styles
sns.set(style="whitegrid", context="notebook", palette="deep")
methods = ['blackhole', 'stratified']
models = df['Model'].unique()
# Create color and style combinations for each method-model pair
colors = sns.color_palette("deep", n_colors=len(models) * len(methods))
method_model_combinations = list(itertools.product(methods, models))
style_dict = {f"{method}_{model}": {
    'color': colors[i],
    'linestyle': '-' if method == 'blackhole' else '--',
    'marker': 'o' if method == 'blackhole' else '^'
} for i, (method, model) in enumerate(method_model_combinations)}

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

# Ensure output folder exists
output_dir = "plots_all_methods_gnns"
os.makedirs(output_dir, exist_ok=True)

# Plotting function to combine all methods and models in one plot per metric
def plot_and_save_combined(metric, std_metric, title_prefix, is_network_metric=False):
    plt.figure(figsize=(12, 6))
    for method, model in method_model_combinations:
        subset = df[(df['Method'] == method) & (df['Model'] == model)].sort_values('Threshold')
        if subset.empty:
            continue
        plt.errorbar(subset['Threshold'], subset[metric],
                     yerr=subset[std_metric],
                     label=f"{method.title()} ({model})",
                     color=style_dict[f"{method}_{model}"]['color'],
                     linestyle=style_dict[f"{method}_{model}"]['linestyle'],
                     marker=style_dict[f"{method}_{model}"]['marker'],
                     capsize=4, markersize=6)

    if metric in ['Accuracy_Mean', 'Cohen_Kappa_Mean']:
        strat_data = df[(df['Method'] == 'stratified') & (df['Model'] == models[0])].set_index('Threshold')
        if 0.0 in strat_data.index:
            baseline = strat_data.loc[0.0, metric]
            plt.axhline(y=baseline, color='gray', linestyle=':', label=f'Baseline (0.0) = {baseline:.3f}')

    metric_name = metric.replace("_Mean", "").replace("_", " ").title()
    plt.title(f'{title_prefix}: {metric_name}', fontsize=14)
    plt.xlabel('Pruning Rate', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(threshold_values)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=10, loc='best', ncol=2)
    plt.tight_layout()

    filename = f"{output_dir}/{title_prefix}_{metric_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# Plot and save performance metrics (all models and methods)
for metric, std_metric in performance_metrics:
    plot_and_save_combined(metric, std_metric, 'Performance')

# Plot and save network metrics (GAT only, both methods)
for metric, std_metric in network_metrics:
    plot_and_save_combined(metric, std_metric, 'Network', is_network_metric=True)