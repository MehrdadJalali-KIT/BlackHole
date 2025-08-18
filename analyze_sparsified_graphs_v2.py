import os
import pandas as pd
import numpy as np
import networkx as nx
import logging
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import modularity
from tqdm import tqdm
from collections import Counter

# Configure logging
logging.basicConfig(
    filename='network_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable debug mode for detailed logging
DEBUG_MODE = True

def compute_degree_distribution(graph):
    """Compute the degree distribution of the graph."""
    try:
        degrees = [degree for _, degree in graph.degree()]
        degree_counts = Counter(degrees)
        degree_dist = {k: v / graph.number_of_nodes() for k, v in degree_counts.items()}
        return degree_dist
    except Exception as e:
        logger.error(f"Error computing degree distribution: {e}")
        return {}

def compute_gravity(graph):
    """Compute gravity metric (sum of edge weights / degree for each node, then average)."""
    try:
        gravity = 0.0
        count = 0
        for node in graph.nodes():
            degree = graph.degree(node, weight='weight')
            if degree > 0:
                total_weight = sum(data['weight'] for _, _, data in graph.edges(node, data=True))
                gravity += total_weight / degree
                count += 1
        return gravity / count if count > 0 else 0.0
    except Exception as e:
        logger.error(f"Error computing gravity: {e}")
        return 0.0

def compute_normalized_graph_energy(graph):
    """Compute normalized graph energy based on Laplacian eigenvalues."""
    try:
        if graph.number_of_nodes() == 0:
            return 0.0
        laplacian = nx.laplacian_matrix(graph).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        energy = sum(abs(eigenvalues))
        return energy / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0.0
    except Exception as e:
        logger.error(f"Error computing normalized graph energy: {e}")
        return 0.0

def compute_unreachable_ratio(graph):
    """Compute ratio of node pairs that are unreachable."""
    try:
        if graph.number_of_nodes() == 0:
            return 1.0
        unreachable_pairs = 0
        total_pairs = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
        for source in graph.nodes():
            paths = nx.single_source_shortest_path_length(graph, source)
            unreachable_pairs += graph.number_of_nodes() - 1 - len(paths) + 1  # Exclude self
        return unreachable_pairs / total_pairs if total_pairs > 0 else 1.0
    except Exception as e:
        logger.error(f"Error computing unreachable ratio: {e}")
        return 1.0

def compute_network_parameters(graph, initial_num_nodes, initial_num_edges, communities):
    """Compute network parameters for a graph."""
    try:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Node and Edge Retention
        percent_nodes_retained = (num_nodes / initial_num_nodes * 100) if initial_num_nodes > 0 else 0.0
        percent_edges_retained = (num_edges / initial_num_edges * 100) if initial_num_edges > 0 else 0.0
        percent_edges_removed = 100.0 - percent_edges_retained if initial_num_edges > 0 else 0.0
        
        # Global Graph Structure
        graph_density = nx.density(graph)
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
        
        # Degree Distribution
        degree_dist = compute_degree_distribution(graph)
        
        # Centrality Metrics
        try:
            degree_centrality = nx.degree_centrality(graph)
            mean_degree_centrality = np.mean(list(degree_centrality.values())) if degree_centrality else 0.0
        except Exception as e:
            mean_degree_centrality = 0.0
            logger.error(f"Error computing degree centrality: {e}")
        
        try:
            betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
            mean_betweenness_centrality = np.mean(list(betweenness_centrality.values())) if betweenness_centrality else 0.0
        except Exception as e:
            mean_betweenness_centrality = 0.0
            logger.error(f"Error computing betweenness centrality: {e}")
        
        # Gravity
        mean_gravity = compute_gravity(graph)
        
        # Normalized Graph Energy
        normalized_graph_energy = compute_normalized_graph_energy(graph)
        
        # Unreachable Ratio
        unreachable_ratio = compute_unreachable_ratio(graph)
        
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
            
            # Community Structure
            try:
                communities_subset = [c & set(subgraph.nodes()) for c in communities if c & set(subgraph.nodes())]
                if communities_subset:
                    modularity_score = modularity(subgraph, communities_subset, weight='weight')
                    mean_community_size = sum(len(c) for c in communities_subset) / len(communities_subset) if communities_subset else 0.0
                    logger.info(f"Computed modularity: {modularity_score} with {len(communities_subset)} communities in subgraph")
                else:
                    communities_subset = nx_comm.louvain_communities(subgraph, seed=42, resolution=0.5)
                    if communities_subset:
                        modularity_score = modularity(subgraph, communities_subset, weight='weight')
                        mean_community_size = sum(len(c) for c in communities_subset) / len(communities_subset) if communities_subset else 0.0
                        logger.info(f"Recomputed {len(communities_subset)} communities, modularity: {modularity_score}")
                    else:
                        modularity_score = 0.0
                        mean_community_size = 0.0
                        logger.warning("No communities detected in subgraph")
            except nx.NetworkXError as e:
                modularity_score = 0.0
                mean_community_size = 0.0
                logger.error(f"Error computing modularity: {e}")
        else:
            diameter = float('inf')
            avg_path_length = float('inf')
            avg_clustering = 0.0
            transitivity = 0.0
            modularity_score = 0.0
            mean_community_size = 0.0
            logger.warning(f"Insufficient nodes ({num_nodes}) or edges ({num_edges}) for advanced metrics")
        
        return {
            'Percent_Nodes_Retained': percent_nodes_retained,
            'Percent_Edges_Retained': percent_edges_retained,
            'Percent_Edges_Removed': percent_edges_removed,
            'Graph_Diameter': diameter,
            'Average_Path_Length': avg_path_length,
            'Graph_Density': graph_density,
            'Average_Degree': avg_degree,
            'Average_Clustering': avg_clustering,
            'Transitivity': transitivity,
            'Modularity': modularity_score,
            'Mean_Community_Size': mean_community_size,
            'Degree_Distribution': degree_dist,
            'Mean_Degree_Centrality': mean_degree_centrality,
            'Mean_Betweenness_Centrality': mean_betweenness_centrality,
            'Mean_Gravity': mean_gravity,
            'Normalized_Graph_Energy': normalized_graph_energy,
            'Unreachable_Ratio': unreachable_ratio,
            'Num_Nodes': num_nodes,
            'Num_Edges': num_edges
        }
    except Exception as e:
        logger.error(f"Error computing network parameters: {e}")
        return {
            'Percent_Nodes_Retained': 0.0,
            'Percent_Edges_Retained': 0.0,
            'Percent_Edges_Removed': 0.0,
            'Graph_Diameter': float('inf'),
            'Average_Path_Length': float('inf'),
            'Graph_Density': 0.0,
            'Average_Degree': 0.0,
            'Average_Clustering': 0.0,
            'Transitivity': 0.0,
            'Modularity': 0.0,
            'Mean_Community_Size': 0.0,
            'Degree_Distribution': {},
            'Mean_Degree_Centrality': 0.0,
            'Mean_Betweenness_Centrality': 0.0,
            'Mean_Gravity': 0.0,
            'Normalized_Graph_Energy': 0.0,
            'Unreachable_Ratio': 1.0,
            'Num_Nodes': 0,
            'Num_Edges': 0
        }

def save_step_results(metrics, threshold, method, run, output_dir="GraphParameters"):
    """Save results for a specific threshold, method, and run."""
    try:
        step_dir = os.path.join(output_dir, f"threshold_{threshold:.2f}", f"method_{method}", f"run_{run}")
        os.makedirs(step_dir, exist_ok=True)
        step_file = os.path.join(step_dir, "parameters.csv")
        # Save degree distribution separately
        degree_dist_file = os.path.join(step_dir, "degree_distribution.csv")
        degree_dist_df = pd.DataFrame([
            {'Degree': k, 'Probability': v}
            for k, v in metrics.pop('Degree_Distribution', {}).items()
        ])
        degree_dist_df.to_csv(degree_dist_file, index=False)
        logger.info(f"Saved degree distribution to {degree_dist_file}")
        # Save other metrics
        df = pd.DataFrame([metrics])
        df = df[[
            'Threshold', 'Method', 'Run', 'Num_Nodes', 'Num_Edges',
            'Percent_Nodes_Retained', 'Percent_Edges_Retained', 'Percent_Edges_Removed',
            'Graph_Diameter', 'Average_Path_Length', 'Graph_Density', 'Average_Degree',
            'Average_Clustering', 'Transitivity', 'Modularity', 'Mean_Community_Size',
            'Mean_Degree_Centrality', 'Mean_Betweenness_Centrality', 'Mean_Gravity',
            'Normalized_Graph_Energy', 'Unreachable_Ratio'
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
        # Collect all results for this method, excluding degree distribution
        method_results = []
        for r in results:
            if r['Method'] == method:
                r_copy = r.copy()
                r_copy.pop('Degree_Distribution', None)  # Remove degree distribution
                method_results.append(r_copy)
        df = pd.DataFrame(method_results)
        if not df.empty:
            df = df[[
                'Threshold', 'Method', 'Run', 'Num_Nodes', 'Num_Edges',
                'Percent_Nodes_Retained', 'Percent_Edges_Retained', 'Percent_Edges_Removed',
                'Graph_Diameter', 'Average_Path_Length', 'Graph_Density', 'Average_Degree',
                'Average_Clustering', 'Transitivity', 'Modularity', 'Mean_Community_Size',
                'Mean_Degree_Centrality', 'Mean_Betweenness_Centrality', 'Mean_Gravity',
                'Normalized_Graph_Energy', 'Unreachable_Ratio'
            ]]
            df.to_csv(method_file, index=False)
            logger.info(f"Saved method results to {method_file}")
        else:
            logger.warning(f"No results to save for method {method}")
        # Save degree distributions for each threshold
        for threshold in set(r['Threshold'] for r in results if r['Method'] == method):
            degree_dist_file = os.path.join(method_dir, f"degree_distribution_t{threshold:.2f}.csv")
            degree_dists = []
            for r in results:
                if r['Method'] == method and r['Threshold'] == threshold:
                    for degree, prob in r.get('Degree_Distribution', {}).items():
                        degree_dists.append({
                            'Run': r['Run'],
                            'Degree': degree,
                            'Probability': prob
                        })
            if degree_dists:
                degree_dist_df = pd.DataFrame(degree_dists)
                degree_dist_df.to_csv(degree_dist_file, index=False)
                logger.info(f"Saved degree distribution for threshold {threshold:.2f} to {degree_dist_file}")
    except Exception as e:
        logger.error(f"Failed to save method results to {method_file}: {e}")

def main():
    methods = ["blackhole", "stratified", "pagerank", "kcenter"]
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
        
    except Exception as e:
        logger.error(f"Failed to load original graph or compute communities: {e}")
        return
    
    # Collect results and track processed combinations
    results = []
    processed_combinations = set()
    missing_combinations = set()
    
    # Total iterations for progress bar
    total_iterations = len(thresholds) * len(methods)
    
    # Progress bar for thresholds and methods
    with tqdm(total=total_iterations, desc="Processing Graphs", position=0) as pbar:
        for threshold in thresholds:
            for method in methods:
                combination = (method, threshold)
                # Construct file path
                base_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_{method}/run_{run}"
                edge_file = f"BH_edges_t{threshold:.2f}_r{run}.csv" if method == 'blackhole' else f"edges_t{threshold:.2f}_r{run}.csv"
                edge_path = os.path.join(base_dir, edge_file)
                
                if not os.path.exists(edge_path):
                    logger.warning(f"File not found: {edge_path}")
                    missing_combinations.add(combination)
                    if DEBUG_MODE:
                        logger.debug(f"Missing combination: {combination} - File: {edge_path}")
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
                        'Run': run
                    })
                    results.append(metrics)
                    processed_combinations.add(combination)
                    
                    # Save step results
                    save_step_results(metrics, threshold, method, run, output_dir)
                    
                except Exception as e:
                    logger.error(f"Failed to process {edge_path}: {e}")
                    missing_combinations.add(combination)
                    if DEBUG_MODE:
                        logger.debug(f"Processing failed for combination: {combination} - Error: {e}")
                
                pbar.update(1)
        
        # Save method results
        for method in methods:
            save_method_results(results, method, output_dir)
    
    # Save final aggregated results
    if results:
        # Save degree distributions separately
        for method in methods:
            for threshold in thresholds:
                degree_dist_file = os.path.join(output_dir, f"degree_distribution_{method}_t{threshold:.2f}.csv")
                degree_dists = []
                for r in results:
                    if r['Method'] == method and r['Threshold'] == threshold:
                        for degree, prob in r.get('Degree_Distribution', {}).items():
                            degree_dists.append({
                                'Run': r['Run'],
                                'Degree': degree,
                                'Probability': prob
                            })
                if degree_dists:
                    degree_dist_df = pd.DataFrame(degree_dists)
                    degree_dist_df.to_csv(degree_dist_file, index=False)
                    logger.info(f"Saved final degree distribution for {method} at threshold {threshold:.2f} to {degree_dist_file}")
        # Save other metrics
        df = pd.DataFrame([r for r in results])
        df = df[[
            'Threshold', 'Method', 'Run', 'Num_Nodes', 'Num_Edges',
            'Percent_Nodes_Retained', 'Percent_Edges_Retained', 'Percent_Edges_Removed',
            'Graph_Diameter', 'Average_Path_Length', 'Graph_Density', 'Average_Degree',
            'Average_Clustering', 'Transitivity', 'Modularity', 'Mean_Community_Size',
            'Mean_Degree_Centrality', 'Mean_Betweenness_Centrality', 'Mean_Gravity',
            'Normalized_Graph_Energy', 'Unreachable_Ratio'
        ]]
        df.to_csv(final_output_file, index=False)
        logger.info(f"Saved final aggregated results to {final_output_file}")
    else:
        logger.error("No results generated")
    
    # Completion report
    total_combinations = set((m, t) for m in methods for t in thresholds)
    processed_count = len(processed_combinations)
    missing_count = len(missing_combinations)
    logger.info(f"Processing completed. Total combinations: {len(total_combinations)}, Processed: {processed_count}, Missing: {missing_count}")
    if missing_combinations:
        logger.info(f"Missing combinations: {missing_combinations}")

if __name__ == "__main__":
    main()