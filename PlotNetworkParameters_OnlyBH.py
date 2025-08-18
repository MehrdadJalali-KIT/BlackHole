import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    filename='plotting.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_plot_style():
    """Configure global plot style using Seaborn and Matplotlib to match GraphSAGE figure."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.facecolor': 'white',
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'lines.markersize': 7,
        'lines.linewidth': 2
    })

def plot_parameter_comparison(df, parameter, output_dir):
    """Create and save a line plot comparing a specific parameter."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        palette = {'blackhole': 'black'}
        styles = {'blackhole': '-'}
        markers = {'blackhole': 'o'}

        for method in df['Method'].unique():
            sub_df = df[df['Method'] == method]
            means = sub_df.groupby('Threshold')[parameter].mean()
            stds = sub_df.groupby('Threshold')[parameter].std()
            thresholds = sorted(means.index)
            ax.errorbar(
                thresholds, means.loc[thresholds], yerr=stds.loc[thresholds],
                label=method.capitalize(), fmt=styles[method]+markers[method], color=palette[method]
            )

        readable_param = parameter.replace("_", " ")
        ax.set_title(readable_param)
        ax.set_xlabel('Threshold (Pruning Rate)')
        ax.set_ylabel(readable_param)
        ax.set_xticks(sorted(df['Threshold'].unique()))
        ax.set_ylim(bottom=0)

        baseline_value = df[df['Threshold'] == 0][parameter].mean()
        ax.axhline(y=baseline_value, linestyle=':', color='gray', linewidth=2.5, label=f"Baseline (original)")

        # Dynamically position legend
        ax.legend(title=None, loc='best')

        plot_path = os.path.join(output_dir, f"{parameter.lower()}_comparison.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved plot to {plot_path}")

    except Exception as e:
        logger.error(f"Failed to plot {parameter}: {e}")

def main():
    input_file = 'GraphParameters/network_parameters_run0.csv'
    output_dir = 'GraphParameters/plots'
    methods = ['blackhole']  # Restricted to blackhole only
    thresholds = [round(t, 1) for t in np.arange(0.9, -0.1, -0.1)]
    run = 0

    setup_plot_style()

    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data: {len(df)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Failed to load {input_file}: {e}")
        return

    df = df[(df['Method'].isin(methods)) & (df['Run'] == 0)].replace([np.inf, -np.inf], np.nan)
    if df.empty:
        logger.error("Filtered DataFrame is empty.")
        return

    parameters = ['Num_Nodes']  # Only plot number of nodes

    for param in parameters:
        if param in df.columns:
            plot_parameter_comparison(df, param, output_dir)
        else:
            logger.warning(f"Parameter not found: {param}")

    logger.info("All plots generated.")

if __name__ == "__main__":
    main()