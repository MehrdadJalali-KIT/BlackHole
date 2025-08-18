import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and filter data
df = pd.read_csv('results_regression_aggregated.csv')
df = df[df['Method'].isin(['blackhole', 'stratified', 'pagerank', 'kcenter'])]
df['MAE_sem'] = df['MAE_Std'] / np.sqrt(1)
df['RMSE_sem'] = df['RMSE_Std'] / np.sqrt(1)
df['R2_sem'] = df['R2_Std'] / np.sqrt(1)

# Define visual styles
sns.set(style="whitegrid", context="talk", palette="deep")  # Larger base style
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
performance_metrics = [('MAE_Mean', 'MAE_sem'), ('RMSE_Mean', 'RMSE_sem'), ('R2_Mean', 'R2_sem')]
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
output_dir = "plots_regression_results"
os.makedirs(output_dir, exist_ok=True)

# Plotting function
def plot_and_save_clean(subset, metric, std_metric, title_prefix):
    plt.figure(figsize=(10, 6))
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
                     capsize=4, markersize=8, linewidth=2)

    if metric in ['MAE_Mean', 'RMSE_Mean', 'R2_Mean']:
        strat_data = subset[subset['Method'] == 'stratified'].set_index('Threshold')
        if 0.0 in strat_data.index:
            baseline = strat_data.loc[0.0, metric]
            plt.axhline(y=baseline, color='gray', linestyle=':', linewidth=1.5,
                        label=f'Baseline (0.0) = {baseline:.3f}')

    # Keep MAE, RMSE, R² capitalized
    metric_name = metric.replace("_Mean", "").replace("_", " ")
    metric_name = metric_name.upper() if metric_name in ['MAE', 'RMSE'] else metric_name

    plt.title(f'{title_prefix}: {metric_name}', fontsize=20, fontweight='bold')
    plt.xlabel('Pruning Rate', fontsize=18)
    plt.ylabel(metric_name, fontsize=18)
    plt.xticks(threshold_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=14)
    plt.tight_layout()

    filename = f"{output_dir}/{title_prefix}_{metric_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=600)
    plt.close()

# Plot performance metrics
for model in models:
    subset = df[df['Model'] == model]
    for metric, std_metric in performance_metrics:
        plot_and_save_clean(subset, metric, std_metric, model)

# Plot network metrics (GAT only)
subset = df[df['Model'] == 'GAT']
for metric, std_metric in network_metrics:
    plot_and_save_clean(subset, metric, std_metric, 'Network')
