import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import ast
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    filename='confusion_matrices_plot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, threshold, model, accuracy, kappa, output_dir, class_names):
    """Plot and save a confusion matrix as a heatmap with percentages using a single color theme for cells."""
    try:
        # Normalize confusion matrix by rows (true class totals)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = (cm / row_sums * 100).round(2)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'}, xticklabels=class_names, yticklabels=class_names)
        # Set title based on threshold
        title = f'{model} at Original Graph' if threshold == 0.0 else f'{model} at Threshold {threshold}'
        plt.title(title, fontweight='bold', fontsize=20)
        plt.xlabel('Predicted', fontweight='bold', fontsize=18)
        plt.ylabel('Actual', fontweight='bold', fontsize=18)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'confusion_matrix_blackhole_run0_{model}_threshold_{threshold:.2f}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix plot to {output_file}")
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix for threshold={threshold:.2f}, model={model}: {e}")

def main():
    input_file = 'evaluation_results/final_results.csv'
    output_dir = 'confusion_matrices_plots'
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    models = ['GAT', 'GCN', 'GraphSAGE']
    class_names = ['nonporous', 'small', 'medium', 'large']  # Based on MOFCSD.csv categories

    # Load CSV
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {input_file} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load {input_file}: {e}")
        return

    # Filter for blackhole, run=0, and GNN models
    df_filtered = df[(df['Method'] == 'blackhole') & (df['Run'] == 0) & (df['Model'].isin(models))]
    df_filtered = df_filtered.drop_duplicates(subset=['Threshold', 'Model', 'Run'])
    logger.info(f"Filtered to {len(df_filtered)} rows for Method=blackhole, Run=0, Models={models}")

    # Total iterations for progress bar
    total_iterations = len(thresholds) * len(models)

    # Plot confusion matrices
    with tqdm(total=total_iterations, desc="Plotting Confusion Matrices", position=0) as pbar:
        for threshold in thresholds:
            for model in models:
                # Find matching row
                row = df_filtered[(df_filtered['Threshold'] == threshold) & (df_filtered['Model'] == model)]

                if row.empty:
                    logger.warning(f"No data for threshold={threshold:.2f}, model={model}")
                    pbar.update(1)
                    continue

                try:
                    # Extract confusion matrix, accuracy, and kappa
                    cm_str = row['Confusion_Matrix'].iloc[0]
                    cm = np.array(ast.literal_eval(cm_str))
                    accuracy = row['Accuracy'].iloc[0]
                    kappa = row['Cohen_Kappa'].iloc[0]

                    # Validate confusion matrix shape
                    if cm.shape != (4, 4):
                        logger.error(f"Invalid confusion matrix shape {cm.shape} for threshold={threshold:.2f}, model={model}")
                        pbar.update(1)
                        continue

                    # Plot confusion matrix
                    plot_confusion_matrix(cm, threshold, model, accuracy, kappa, output_dir, class_names)
                except Exception as e:
                    logger.error(f"Failed to process threshold={threshold:.2f}, model={model}: {e}")

                pbar.update(1)

    logger.info(f"Completed plotting. Check {output_dir} for PNG files.")

if __name__ == "__main__":
    main()