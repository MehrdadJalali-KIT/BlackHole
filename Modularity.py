import os
import pandas as pd
import numpy as np
import networkx as nx
import logging
import networkx.algorithms.community as nx_comm
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    filename='modularity_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_modularity(graph, method, threshold):
    """Compute modularity for a graph with and without weights."""
    try:
        # Recompute communities on the subgraph with resolution 1.0
        communities_subset = nx_comm.louvain_communities(graph, seed=42, resolution=1.0)
        if not communities_subset:
            logger.warning(f"No communities detected for {method} at threshold {threshold:.2f}")
            return {'Method': method, 'Threshold': threshold, 'Modularity_Weighted': 0.0, 'Modularity_Unweighted': 0.0}

        # Calculate modularity with weights
        modularity_score_weighted = nx_comm.modularity(graph, communities_subset, weight='weight')
        
        # Calculate modularity without weights
        modularity_score_unweighted = nx_comm.modularity(graph, communities_subset)

        logger.info(f"Computed modularity for {method} at threshold {threshold:.2f}: Weighted={modularity_score_weighted}, Unweighted={modularity_score_unweighted}")
        return {
            'Method': method,
            'Threshold': threshold,
            'Modularity_Weighted': modularity_score_weighted,
            'Modularity_Unweighted': modularity_score_unweighted
        }
    except Exception as e:
        logger.error(f"Error computing modularity for {method} at threshold {threshold:.2f}: {e}")
        return {'Method': method, 'Threshold': threshold, 'Modularity_Weighted': 0.0, 'Modularity_Unweighted': 0.0}

def main():
    methods = ['blackhole', 'stratified']
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    run = 0
    output_dir = 'GraphParameters'
    output_file = os.path.join(output_dir, 'modularity_results.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results
    results = []
    
    # Total iterations for progress bar
    total_iterations = len(thresholds) * len(methods)
    
    # Progress bar for thresholds and methods
    with tqdm(total=total_iterations, desc="Processing Modularity", position=0) as pbar:
        for threshold in thresholds:
            for method in methods:
                # Construct file path
                base_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_{method}/run_{run}"
                edge_file = f"BH_edges_t{threshold:.2f}_r{run}.csv" if method == 'blackhole' else f"edges_t{threshold:.2f}_r{run}.csv"
                edge_path = os.path.join(base_dir, edge_file)
                
                if not os.path.exists(edge_path):
                    logger.warning(f"File not found: {edge_path}")
                    pbar.update(1)
                    continue
                
                try:
                    # Load sparsified graph
                    sparse_edges = pd.read_csv(edge_path)
                    graph = nx.from_pandas_edgelist(sparse_edges, 'source', 'target', 'weight')
                    logger.info(f"Loaded {method} graph (threshold={threshold:.2f}, run={run}): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                    
                    # Compute modularity
                    metrics = compute_modularity(graph, method, threshold)
                    results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Failed to process {edge_path}: {e}")
                
                pbar.update(1)
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved modularity results to {output_file}")
    else:
        logger.error("No modularity results generated")

if __name__ == "__main__":
    main()