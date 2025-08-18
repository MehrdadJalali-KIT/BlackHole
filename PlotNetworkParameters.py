import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import Counter
import networkx as nx
from itertools import cycle

# ----------------------- Config ---------------------------------
# Map methods to their edge-file name patterns (without directories).
# Customize here if your filenames differ.
EDGE_FILE_PATTERNS = {
    "blackhole": "BH_edges_t{threshold:.2f}_r{run}.csv",
    "stratified": "edges_t{threshold:.2f}_r{run}.csv",
    "pagerank": "PR_edges_t{threshold:.2f}_r{run}.csv",
    "kcenter": "KC_edges_t{threshold:.2f}_r{run}.csv",
}
# ----------------------------------------------------------------

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

def _style_cyclers(methods):
    """Create deterministic palettes/markers/linestyles for an arbitrary set of methods."""
    base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not base_colors:
        base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                       "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    color_cycle = cycle(base_colors)

    # Helpful defaults; override known methods for consistency
    default_markers = cycle(['o','^','s','D','X','P','v','>','<','h'])
    default_styles  = cycle(['-','--','-.',':'])

    palette = {}
    markers = {}
    styles  = {}

    # Preferred consistent mapping for common methods
    preferred = {
        'blackhole': ('black', 'o', '-'),
        'stratified': ('#2ca02c', '^', '--'),
        'pagerank': ('#1f77b4', 's', '-'),
        'kcenter': ('#d62728', 'D', '--'),
    }

    for m in methods:
        key = str(m).lower()
        if key in preferred:
            palette[m], markers[m], styles[m] = preferred[key]
        else:
            palette[m] = next(color_cycle)
            markers[m] = next(default_markers)
            styles[m]  = next(default_styles)

    return palette, markers, styles

def plot_parameter_comparison(df, parameter, output_dir):
    """Create and save a line plot comparing a specific parameter for all methods present in df."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        methods = list(df['Method'].unique())
        palette, markers, styles = _style_cyclers(methods)

        for method in methods:
            sub_df = df[df['Method'] == method]
            if sub_df.empty or parameter not in sub_df.columns:
                logger.warning(f"No data for method={method}, parameter={parameter}")
                continue

            grouped = sub_df.groupby('Threshold')[parameter]
            means = grouped.mean()
            stds = grouped.std()
            thresholds = sorted(means.index)

            # Build a format string combining style and marker (safe fallback if unknown)
            fmt = f"{styles.get(method,'-')}{markers.get(method,'o')}"
            ax.errorbar(
                thresholds,
                means.loc[thresholds],
                yerr=stds.loc[thresholds],
                label=method.capitalize(),
                fmt=fmt,
                color=palette.get(method, None),
                capsize=3,
                elinewidth=1.2
            )

        readable_param = parameter.replace("_", " ")
        ax.set_title(readable_param)
        ax.set_xlabel('Threshold (Pruning Rate)')
        ax.set_ylabel(readable_param)
        ax.set_xticks(sorted(df['Threshold'].unique()))
        # Only force bottom=0 when the data is strictly nonnegative
        if (df[parameter] >= 0).all():
            ax.set_ylim(bottom=0)

        # Baseline (original) at Threshold == 0 if available
        base = df[df['Threshold'] == 0]
        if not base.empty:
            baseline_value = base[parameter].mean()
            ax.axhline(
                y=baseline_value,
                linestyle=':',
                color='gray',
                linewidth=2.0,
                label="Baseline (original)"
            )

        ax.legend(title=None, loc='best')
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"{parameter.lower()}_comparison.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved plot to {plot_path}")

    except Exception as e:
        logger.error(f"Failed to plot {parameter}: {e}")

def compute_degree_distribution(graph: nx.Graph):
    """
    Return a Counter mapping degree -> count.
    Handles isolated nodes correctly.
    """
    try:
        degrees = [d for _, d in graph.degree()]
        return Counter(degrees)
    except Exception as e:
        logger.error(f"compute_degree_distribution failed: {e}")
        return Counter()

def create_3d_histogram(degree_data_by_threshold, method, thresholds, output_dir):
    """
    Create a 3D bar plot of degree distributions across thresholds for a specific method.

    degree_data_by_threshold: dict[threshold -> Counter(degree -> count)]
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Collect all degrees that appear across thresholds
        all_degrees = sorted(set().union(*[set(cnt.keys()) for cnt in degree_data_by_threshold.values()]))
        if not all_degrees:
            logger.warning(f"No degrees to plot for method={method}")
            return

        # Build matrices (Y: thresholds idx, X: degree bins)
        thresholds_present = sorted(degree_data_by_threshold.keys())
        X = np.arange(len(all_degrees))
        Y = np.arange(len(thresholds_present))

        Z = np.zeros((len(Y), len(X)), dtype=float)
        for yi, t in enumerate(thresholds_present):
            cnt = degree_data_by_threshold[t]
            for xi, deg in enumerate(all_degrees):
                Z[yi, xi] = cnt.get(deg, 0)

        # 3D bar plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        xpos, ypos = np.meshgrid(X, Y, indexing='xy')
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)

        dx = 0.6 * np.ones_like(zpos)
        dy = 0.6 * np.ones_like(zpos)
        dz = Z.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

        ax.set_xlabel('Degree')
        ax.set_ylabel('Threshold Index')
        ax.set_zlabel('Count')

        ax.set_xticks(np.arange(len(all_degrees)))
        ax.set_xticklabels([str(d) for d in all_degrees], rotation=45, ha='right')

        # Map threshold index to actual threshold in y tick labels
        ax.set_yticks(np.arange(len(thresholds_present)))
        ax.set_yticklabels([f"{t:.1f}" for t in thresholds_present])

        ax.set_title(f"Degree Distribution (3D) — {method.capitalize()}")
        fig.tight_layout()

        outpath = os.path.join(output_dir, f"degree_3d_{method.lower()}.png")
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved 3D histogram to {outpath}")
    except Exception as e:
        logger.error(f"create_3d_histogram failed for {method}: {e}")

def main():
    # ------- I/O & selection -------
    input_file = 'GraphParameters/network_parameters_run0.csv'
    output_dir = 'GraphParameters/plots'

    # Add new methods here; plotting & file-loading will adapt automatically
    methods = ['blackhole', 'stratified', 'pagerank', 'kcenter']

    thresholds = [round(t, 1) for t in np.arange(0.9, -0.1, -0.1)]
    run = 0

    setup_plot_style()

    # ------- Load table of precomputed parameters -------
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data: {len(df)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Failed to load {input_file}: {e}")
        return

    # Normalize method names to lowercase for consistency
    if 'Method' in df.columns:
        df['Method'] = df['Method'].astype(str).str.lower()
    else:
        logger.error("Input CSV must contain a 'Method' column.")
        return

    if 'Run' not in df.columns:
        logger.error("Input CSV must contain a 'Run' column.")
        return

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Keep only desired methods and specific run
    df = df[(df['Method'].isin(methods)) & (df['Run'] == run)]
    if df.empty:
        logger.error("Filtered DataFrame is empty after method/run selection.")
        return

    # ------- Parameters to plot -------
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
            logger.warning(f"Parameter not found in CSV: {param}")

    # ------- Degree distribution 3D plots from per-threshold edge files -------
    for method in methods:
        degree_data = {}
        for threshold in thresholds:
            base_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_{method}/run_{run}"
            pattern = EDGE_FILE_PATTERNS.get(method, "edges_t{threshold:.2f}_r{run}.csv")
            edge_file = pattern.format(threshold=threshold, run=run)
            edge_path = os.path.join(base_dir, edge_file)

            if not os.path.exists(edge_path):
                logger.warning(f"File not found: {edge_path}")
                continue

            try:
                sparse_edges = pd.read_csv(edge_path)
                # Expect 'source','target' columns; rename common variants
                cols = {c.lower(): c for c in sparse_edges.columns}
                src_col = 'source' if 'source' in cols else ('src' if 'src' in cols else None)
                dst_col = 'target' if 'target' in cols else ('dst' if 'dst' in cols else None)
                if src_col is None or dst_col is None:
                    # Fallback: try to guess first two columns
                    src_col, dst_col = sparse_edges.columns[:2]
                    logger.warning(f"Using first two columns as source/target for {edge_path}")

                graph = nx.from_pandas_edgelist(sparse_edges, src_col, dst_col, edge_attr=True)
                logger.info(f"Loaded {method} graph (threshold={threshold:.2f}, run={run}): "
                            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
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
