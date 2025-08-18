import os
import pandas as pd
import numpy as np
import networkx as nx
import logging
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import modularity
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    filename='network_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_network_parameters(graph, initial_num_nodes, initial_num_edges, communities):
    """Compute network parameters for a graph."""
    try:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Node and Edge Retention
        percent_nodes_retained = (num_nodes / initial_num_nodes * 100) if initial_num_nodes > 0 else 0.0
        percent_edges_retained = (num_edges / initial_num_edges * 100) if initial_num_edges > 0 else 0.0
        
        # Global Graph Structure
        graph_density = nx.density(graph)
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
        
        # Use largest connected component for path-based metrics
        if num_nodes > 1 and num_edges > 0:
            largest_cc = max(nx.connected_components(graph), key=len, default=set())
            subgraph = graph.subgraph(largest_cc).copy()
            logger.info(f"Using largest connected component: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
            
            # Graph Diameter and Average Path Length
            try:
                if nx.is_connected(subgraph):
                    diameter = nx.diameter(subgraph, weight='weight')
                    avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
                else:
                    diameter = float('inf')
                    avg_path_length = float('inf')
                    logger.warning("Graph is disconnected, setting diameter and path length to infinity")
            except nx.NetworkXError as e:
                diameter = float('inf')
                avg_path_length = float('inf')
                logger.error(f"Error computing diameter/path length: {e}")
            
            # Local Neighborhood Structure
            try:
                avg_clustering = nx.average_clustering(subgraph, weight='weight')
                transitivity = nx.transitivity(subgraph)
            except nx.NetworkXError as e:
                avg_clustering = 0.0
                transitivity = 0.0
                logger.error(f"Error computing clustering/transitivity: {e}")
            
            # Modularity
            try:
                # Use original communities, restricted to subgraph nodes
                communities_subset = [c & set(subgraph.nodes()) for c in communities if c & set(subgraph.nodes())]
                if communities_subset:
                    modularity_score = modularity(subgraph, communities_subset, weight='weight')
                    logger.info(f"Computed modularity: {modularity_score} with {len(communities_subset)} communities in subgraph")
                else:
                    # Fallback: recompute communities
                    communities_subset = nx_comm.louvain_communities(subgraph, seed=42, resolution=0.5)
                    if communities_subset:
                        modularity_score = modularity(subgraph, communities_subset, weight='weight')
                        logger.info(f"Recomputed {len(communities_subset)} communities, modularity: {modularity_score}")
                    else:
                        modularity_score = 0.0
                        logger.warning("No communities detected in subgraph")
            except nx.NetworkXError as e:
                modularity_score = 0.0
                logger.error(f"Error computing modularity: {e}")
        else:
            diameter = float('inf')
            avg_path_length = float('inf')
            avg_clustering = 0.0
            transitivity = 0.0
            modularity_score = 0.0
            logger.warning(f"Insufficient nodes ({num_nodes}) or edges ({num_edges}) for advanced metrics")
        
        return {
            'Percent_Nodes_Retained': percent_nodes_retained,
            'Percent_Edges_Retained': percent_edges_retained,
            'Graph_Diameter': diameter,
            'Average_Path_Length': avg_path_length,
            'Graph_Density': graph_density,
            'Average_Degree': avg_degree,
            'Average_Clustering': avg_clustering,
            'Transitivity': transitivity,
            'Modularity': modularity_score
        }
    except Exception as e:
        logger.error(f"Error computing network parameters: {e}")
        return {
            'Percent_Nodes_Retained': 0.0,
            'Percent_Edges_Retained': 0.0,
            'Graph_Diameter': float('inf'),
            'Average_Path_Length': float('inf'),
            'Graph_Density': 0.0,
            'Average_Degree': 0.0,
            'Average_Clustering': 0.0,
            'Transitivity': 0.0,
            'Modularity': 0.0
        }

def save_step_results(metrics, threshold, method, run, output_dir="GraphParameters"):
    """Save results for a specific threshold, method, and run."""
    try:
        step_dir = os.path.join(output_dir, f"threshold_{threshold:.2f}", f"method_{method}", f"run_{run}")
        os.makedirs(step_dir, exist_ok=True)
        step_file = os.path.join(step_dir, "parameters.csv")
        df = pd.DataFrame([metrics])
        df = df[[
            'Threshold', 'Method', 'Run', 'Num_Nodes', 'Num_Edges',
            'Percent_Nodes_Retained', 'Percent_Edges_Retained',
            'Graph_Diameter', 'Average_Path_Length', 'Graph_Density', 'Average_Degree',
            'Average_Clustering', 'Transitivity', 'Modularity', 'Average_Community_Size'
        ]]
        df.to_csv(step_file, index=False)
        logger.info(f"Saved step results to {step_file}")
    except Exception as e:
        logger.error(f"Failed to save step results to {step_file}: {e}")

def save_method_results(results, method, output_dir="GraphParameters"):
    """Save aggregated results for a specific method."""
    try:
        method_dir = os.path.join(output_dir, f"method_{method}")
        os.makedirs(method_dir, exist_ok=True)
        method_file = os.path.join(method_dir, "parameters.csv")
        df = pd.DataFrame([r for r in results if r['Method'] == method])
        if not df.empty:
            df = df[[
                'Threshold', 'Method', 'Run', 'Num_Nodes', 'Num_Edges',
                'Percent_Nodes_Retained', 'Percent_Edges_Retained',
                'Graph_Diameter', 'Average_Path_Length', 'Graph_Density', 'Average_Degree',
                'Average_Clustering', 'Transitivity', 'Modularity', 'Average_Community_Size'
            ]]
            df.to_csv(method_file, index=False)
            logger.info(f"Saved method results to {method_file}")
        else:
            logger.warning(f"No results to save for method {method}")
    except Exception as e:
        logger.error(f"Failed to save method results to {method_file}: {e}")

def main():
    methods = ['blackhole', 'stratified']
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    run = 0
    edges_list_filename = 'MOFGalaxyNet.csv'
    output_dir = 'GraphParameters'
    final_output_file = os.path.join(output_dir, 'network_parameters_run0.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original graph and compute communities
    try:
        edges_list = pd.read_csv(edges_list_filename)
        original_graph = nx.from_pandas_edgelist(edges_list, 'source', 'target', 'weight')
        initial_num_nodes = original_graph.number_of_nodes()
        initial_num_edges = original_graph.number_of_edges()
        logger.info(f"Original graph: {initial_num_nodes} nodes, {initial_num_edges} edges")
        
        # Compute original communities
        communities = nx_comm.louvain_communities(original_graph, seed=42, resolution=0.5)
        logger.info(f"Detected {len(communities)} communities in original graph")
        
        # Compute average community size
        avg_community_size = sum(len(c) for c in communities) / len(communities) if communities else 0.0
        logger.info(f"Average community size: {avg_community_size}")
    except Exception as e:
        logger.error(f"Failed to load original graph or compute communities: {e}")
        return
    
    # Collect results
    results = []
    
    # Total iterations for progress bar
    total_iterations = len(thresholds) * len(methods)
    
    # Progress bar for thresholds and methods
    with tqdm(total=total_iterations, desc="Processing Graphs", position=0) as pbar:
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
                    
                    # Compute network parameters
                    metrics = compute_network_parameters(graph, initial_num_nodes, initial_num_edges, communities)
                    metrics.update({
                        'Threshold': threshold,
                        'Method': method,
                        'Run': run,
                        'Num_Nodes': graph.number_of_nodes(),
                        'Num_Edges': graph.number_of_edges(),
                        'Average_Community_Size': avg_community_size
                    })
                    results.append(metrics)
                    
                    # Save step results
                    save_step_results(metrics, threshold, method, run, output_dir)
                    
                except Exception as e:
                    logger.error(f"Failed to process {edge_path}: {e}")
                
                pbar.update(1)
        
        # Save method results
        for method in methods:
            save_method_results(results, method, output_dir)
    
    # Save final aggregated results
    if results:
        df = pd.DataFrame(results)
        df = df[[
            'Threshold', 'Method', 'Run', 'Num_Nodes', 'Num_Edges',
            'Percent_Nodes_Retained', 'Percent_Edges_Retained',
            'Graph_Diameter', 'Average_Path_Length', 'Graph_Density', 'Average_Degree',
            'Average_Clustering', 'Transitivity', 'Modularity', 'Average_Community_Size'
        ]]
        df.to_csv(final_output_file, index=False)
        logger.info(f"Saved final aggregated results to {final_output_file}")
    else:
        logger.error("No results generated")

if __name__ == "__main__":
    main()