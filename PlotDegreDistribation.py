import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import networkx as nx

# Configure logging
logging.basicConfig(
    filename='plotting.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_plot_style():
    """Configure global plot style using Seaborn and Matplotlib."""
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
        'grid.linestyle': '--'
    })

def plot_parameter_comparison(df, parameter, output_dir):
    """Create and save a line plot comparing a specific parameter."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x='Threshold',
            y=parameter,
            hue='Method',
            style='Method',
            markers=True,
            dashes=False,
            palette='tab10',
            ax=ax
        )

        readable_param = parameter.replace("_", " ")
        ax.set_title(f'{readable_param} by Threshold', pad=20)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(f'Average {readable_param}')
        ax.set_xticks(sorted(df['Threshold'].unique()))
        ax.legend(title='Method', loc='upper right')

        if parameter == 'Average_Degree':
            y_max, y_min = df[parameter].max(), df[parameter].min()
            ax.annotate('Highest', xy=(0.9, y_max), xytext=(0.85, y_max + 2),
                        arrowprops=dict(facecolor='black', shrink=0.05))
            ax.annotate('Lowest', xy=(0.0, y_min), xytext=(0.05, y_min - 2),
                        arrowprops=dict(facecolor='black', shrink=0.05))

        plot_path = os.path.join(output_dir, f"{parameter.lower()}_comparison.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved plot to {plot_path}")

    except Exception as e:
        logger.error(f"Failed to plot {parameter}: {e}")

def compute_degree_distribution(graph):
    """Compute the degree distribution of the graph."""
    try:
        degrees = [degree for _, degree in graph.degree()]
        degree_counts = Counter(degrees)
        return dict(degree_counts)
    except Exception as e:
        logger.error(f"Error computing degree distribution: {e}")
        return {}

def create_3d_histogram(degree_data, method, thresholds, output_dir):
    """Create a 3D bar plot for degree frequencies across thresholds."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        max_degree = max((max(d.keys()) for d in degree_data.values()), default=10)
        hist_bins = np.arange(max_degree + 1)
        hist_data = np.zeros((len(thresholds), len(hist_bins)))

        for i, threshold in enumerate(thresholds):
            if threshold in degree_data:
                for degree, freq in degree_data[threshold].items():
                    if degree < len(hist_bins):
                        hist_data[i, degree] += freq

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        x, y = np.meshgrid(hist_bins, np.array(thresholds))
        z = hist_data.T

        for i in range(len(hist_bins)):
            for j in range(len(thresholds)):
                if z[i, j] > 0:
                    ax.bar3d(x[j, i], y[j, i], 0, 0.8, 0.08, z[i, j], shade=True, color=plt.cm.viridis(z[i, j] / z.max()))

        ax.set_title(f'3D Histogram of Degree Frequencies ({method})')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Threshold')
        ax.set_zlabel('Frequency')
        ax.set_xticks(hist_bins)
        ax.set_yticks(thresholds)
        ax.view_init(elev=20, azim=-60)

        plot_path = os.path.join(output_dir, f"3d_histogram_{method}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved 3D histogram plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create 3D histogram for {method}: {e}")

def main():
    input_file = 'GraphParameters/network_parameters_run0.csv'
    output_dir = 'GraphParameters/plots'
    methods = ['blackhole', 'stratified']
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

    parameters = [
        'Num_Nodes', 'Num_Edges', 'Percent_Nodes_Retained', 'Percent_Edges_Retained',
        'Percent_Edges_Removed', 'Graph_Diameter', 'Average_Path_Length', 'Graph_Density',
        'Average_Degree', 'Average_Clustering', 'Transitivity', 'Modularity',
        'Mean_Community_Size', 'Mean_Degree_Centrality', 'Mean_Betweenness_Centrality',
        'Mean_Gravity', 'Normalized_Graph_Energy', 'Unreachable_Ratio'
    ]

    for param in parameters:
        if param in df.columns:
            plot_parameter_comparison(df, param, output_dir)
        else:
            logger.warning(f"Parameter not found: {param}")

    for method in methods:
        degree_data = {}
        for threshold in thresholds:
            base_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_{method}/run_{run}"
            edge_file = f"BH_edges_t{threshold:.2f}_r{run}.csv" if method == 'blackhole' else f"edges_t{threshold:.2f}_r{run}.csv"
            edge_path = os.path.join(base_dir, edge_file)

            if not os.path.exists(edge_path):
                logger.warning(f"File not found: {edge_path}")
                continue

            try:
                sparse_edges = pd.read_csv(edge_path)
                graph = nx.from_pandas_edgelist(sparse_edges, 'source', 'target', edge_attr=True)
                logger.info(f"Loaded {method} graph (threshold={threshold:.2f}, run={run}): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                degree_dist = compute_degree_distribution(graph)
                if degree_dist:
                    degree_data[threshold] = degree_dist
                else:
                    logger.warning(f"No degree distribution computed for {method} at threshold {threshold:.2f}")
            except Exception as e:
                logger.error(f"Failed to process {edge_path}: {e}")

        if degree_data:
            create_3d_histogram(degree_data, method, thresholds, output_dir)

    logger.info("All plots generated.")

if __name__ == "__main__":
    main()