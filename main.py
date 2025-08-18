import os
import time
import logging
import traceback
import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import modularity
from data_utils import load_edges_list, load_summary_data
from bh_sparsification import generate_bootstrapped_blackhole_edges, calculate_gravity_per_community
from experiment_manager import save_results, aggregate_results, save_checkpoint, load_checkpoint
from sparsification_methods import sparsify_edges, save_sparsified_edges
from graphsage_model import GraphSAGE, GCN, GAT, train, test
import psutil
import argparse

# Configure logging
logging.basicConfig(
    filename='bh_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"Memory usage: RSS={mem_info.rss / 1024 ** 2:.2f} MB, VMS={mem_info.vms / 1024 ** 2:.2f} MB")
        return mem_info.rss / 1024 ** 2
    except ImportError:
        logger.warning("psutil not installed, skipping memory usage logging")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sparsification and GNN evaluation")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification',
                        help='Task to perform: classification or regression')
    args = parser.parse_args()
    task = args.task

    start_time = time.time()
    methods = ["blackhole", "stratified", "pagerank", "kcenter"]
    models = ["GAT", "GCN", "GraphSAGE"]
    num_runs = 4
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    fixed_test_nodes_file = 'fixed_test_nodes.csv'
    edges_list_filename = 'MOFGalaxyNet.csv'
    summary_data_filename = 'MOFCSD.csv'
    checkpoint_file = os.path.abspath('bh_evaluation_checkpoint.json')
    debug_mode = False
    use_edge_weights = True
    performance_metrics_file = 'sparsification_performance.csv'

    os.makedirs(os.path.dirname(checkpoint_file) or '.', exist_ok=True)

    logger.info(f"Starting Bootstrapped Black Hole Strategy and Evaluation (Task: {task})")
    log_memory_usage()

    completed_sparsification = set()
    completed_evaluation = set()
    results = []
    performance_metrics = []

    # Load data
    try:
        edges_list = load_edges_list(edges_list_filename)
        if not all(col in edges_list.columns for col in ['source', 'target', 'weight']):
            raise ValueError("MOFGalaxyNet.csv must contain 'source', 'target', 'weight' columns")
        logger.info(f"Loaded MOFGalaxyNet.csv: {len(edges_list)} edges, columns: {list(edges_list.columns)}")
        node_labels = pd.concat([edges_list['source'], edges_list['target']]).unique()
        features_df, summary_data = load_summary_data(summary_data_filename, node_labels)
        logger.info(f"Loaded MOFCSD.csv: {len(summary_data)} nodes, index: {list(summary_data.index[:5])}")
        logger.info(f"Summary data columns: {list(summary_data.columns)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}\n{traceback.format_exc()}")
        exit(1)

    # Validate node consistency
    summary_nodes = set(summary_data.index)
    edge_nodes = set(node_labels)
    missing_nodes = edge_nodes - summary_nodes
    if missing_nodes:
        logger.warning(f"{len(missing_nodes)} nodes in MOFGalaxyNet.csv not in MOFCSD.csv: {list(missing_nodes)[:10]}")

    # Construct initial graph
    graph = nx.Graph()
    try:
        for _, row in edges_list.iterrows():
            graph.add_edge(row['source'], row['target'], weight=row['weight'])
        initial_num_nodes = graph.number_of_nodes()
        initial_num_edges = graph.number_of_edges()
        if initial_num_nodes == 0 or initial_num_edges == 0:
            raise ValueError(f"Initial graph is empty: {initial_num_nodes} nodes, {initial_num_edges} edges")
        logger.info(f"Initial graph: {initial_num_nodes} nodes, {initial_num_edges} edges")
    except Exception as e:
        logger.error(f"Failed to construct initial graph: {e}\n{traceback.format_exc()}")
        exit(1)

    # Perform community detection
    try:
        communities = nx_comm.louvain_communities(graph, seed=42)
        logger.info(f"Detected {len(communities)} communities in original graph")
    except Exception as e:
        logger.error(f"Failed to detect communities: {e}\n{traceback.format_exc()}")
        exit(1)

    # Calculate gravity scores and test nodes
    try:
        gravity, degree_centrality, betweenness_centrality, edge_weight_sum, fixed_test_nodes = calculate_gravity_per_community(graph, communities)
        pd.DataFrame({'node': list(fixed_test_nodes)}).to_csv(fixed_test_nodes_file, index=False)
        logger.info(f"Saved {len(fixed_test_nodes)} fixed test nodes to {fixed_test_nodes_file}")
    except Exception as e:
        logger.error(f"Failed to calculate gravity scores: {e}\n{traceback.format_exc()}")
        exit(1)

    # Calculate PageRank scores
    try:
        pagerank_scores = nx.pagerank(graph, alpha=0.85)
        logger.info(f"Computed PageRank scores for {len(pagerank_scores)} nodes")
    except Exception as e:
        logger.error(f"Failed to compute PageRank scores: {e}\n{traceback.format_exc()}")
        exit(1)

    # Prepare features and labels
    try:
        required_columns = ['Pore Limiting Diameter', 'linker SMILES', 'metal', 'Largest Cavity Diameter', 'Largest Free Sphere']
        if not all(col in summary_data.columns for col in required_columns):
            raise ValueError(f"MOFCSD.csv missing required columns: {required_columns}")
        x = torch.tensor(features_df.values, dtype=torch.float)
        if task == 'classification':
            labels = pd.Series(summary_data['category']).astype('category').cat.codes.to_numpy()
            y = torch.tensor(labels, dtype=torch.long)
            class_counts = np.bincount(labels)
            class_weights = 1.0 / (class_counts + 1e-6) ** 0.5
            class_weights = torch.tensor(class_weights, dtype=torch.float) / class_weights.sum()
            logger.info(f"Classification: Label distribution: {dict(pd.Series(labels).value_counts())}")
            logger.info(f"Classification: Class weights: {class_weights.tolist()}")
        else:  # regression
            y = torch.tensor(summary_data['Pore Limiting Diameter'].values, dtype=torch.float)
            if y.isnan().any():
                median_pld = summary_data['Pore Limiting Diameter'].median()
                y = torch.where(y.isnan(), torch.tensor(median_pld, dtype=torch.float), y)
                logger.warning(f"Null values in 'Pore Limiting Diameter', filled with median: {median_pld:.4f}")
            class_weights = None
            logger.info(f"Regression: PLD range: min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}")
        logger.info(f"Prepared features with shape {x.shape} and labels with shape {y.shape}")
    except Exception as e:
        logger.error(f"Failed to prepare features and labels: {e}\n{traceback.format_exc()}")
        exit(1)

    # Load or reset checkpoint
    try:
        checkpoint = load_checkpoint(checkpoint_file)
        completed_sparsification = set(tuple(x) for x in checkpoint.get("completed_sparsification", []))
        completed_evaluation = set(tuple(x) for x in checkpoint.get("completed_evaluation", []))
        results = checkpoint.get("results", [])
        checkpoint_fixed_test_nodes = set(checkpoint.get("fixed_test_nodes", []))
        if checkpoint_fixed_test_nodes and checkpoint_fixed_test_nodes != fixed_test_nodes:
            logger.warning("Fixed test nodes in checkpoint differ from computed nodes. Using computed nodes.")
        logger.info(f"Loaded checkpoint: {len(completed_sparsification)} sparsifications, {len(completed_evaluation)} evaluations completed")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}\n{traceback.format_exc()}")
        logger.info("Resetting checkpoint due to load failure")
        checkpoint = {
            "completed_sparsification": [],
            "completed_evaluation": [],
            "results": [],
            "fixed_test_nodes": list(fixed_test_nodes)
        }
        save_checkpoint(checkpoint_file, checkpoint)

    if debug_mode:
        thresholds = [0.9]
        num_runs = 1
        runs = [0]
    else:
        runs = range(num_runs)

    total_experiments = len(thresholds) * len(runs) * len(methods) * len(models)
    experiment_counter = 0
    bh_nodes_retained = {}

    with tqdm(total=total_experiments, desc="Overall Progress", position=0) as pbar:
        for threshold in thresholds:
            logger.info(f"Processing threshold: {threshold:.2f}")
            try:
                for method in methods:
                    for run in runs:
                        log_memory_usage()
                        target_num_nodes = int(max(0.2 * initial_num_nodes, (1 - threshold) * initial_num_nodes))
                        target_num_edges = int(max(10, (1 - threshold) * initial_num_edges))
                        logger.info(f"Method {method}, run {run}: Target nodes: {target_num_nodes} ({(1 - threshold) * 100:.2f}% of {initial_num_nodes}), Target edges: {target_num_edges}")

                        # Sparsification
                        sparsification_id = (threshold, run, method)
                        start_sparsification_time = time.time()
                        peak_memory = log_memory_usage()
                        sparse_edges = None
                        nodes_retained = 0

                        if method == "blackhole":
                            sparse_edges, nodes_retained = generate_bootstrapped_blackhole_edges(
                                edges_list, int(0.5 * len(edges_list)), threshold, run,
                                communities, gravity, degree_centrality, betweenness_centrality,
                                edge_weight_sum, fixed_test_nodes, summary_data
                            )
                            bh_nodes_retained[(threshold, run)] = nodes_retained
                        else:
                            sparse_edges = sparsify_edges(
                                edges_list, bh_nodes_retained.get((threshold, run), target_num_nodes),
                                method, list(graph.nodes), target_num_edges, fixed_test_nodes, pagerank_scores
                            )
                            save_sparsified_edges(sparse_edges, threshold, method, run)
                            nodes_retained = len(set(sparse_edges['source']).union(set(sparse_edges['target'])))

                        if sparse_edges.empty:
                            logger.warning(f"No edges after sparsification for method {method}, threshold {threshold}, run {run}")
                            pbar.update(len(models))
                            continue

                        completed_sparsification.add(sparsification_id)
                        sparsification_time = time.time() - start_sparsification_time
                        performance_metrics.append({
                            'Threshold': threshold,
                            'Method': method,
                            'Run': run,
                            'Sparsification_Time': sparsification_time,
                            'Peak_Memory_MB': peak_memory
                        })
                        pd.DataFrame(performance_metrics).to_csv(performance_metrics_file, index=False)

                        # Build graph from sparse edges
                        graph = nx.Graph()
                        valid_nodes = set(summary_data.index).intersection(set(sparse_edges['source']).union(set(sparse_edges['target'])).union(fixed_test_nodes))
                        graph.add_nodes_from(valid_nodes)
                        for _, row in sparse_edges.iterrows():
                            if row['source'] in valid_nodes and row['target'] in valid_nodes:
                                graph.add_edge(row['source'], row['target'], weight=row['weight'])
                        graph_nodes = set(graph.nodes())
                        num_nodes = len(graph_nodes)
                        num_edges = graph.number_of_edges()
                        nodes_removed = initial_num_nodes - num_nodes
                        edges_removed = initial_num_edges - num_edges
                        percent_nodes_retained = num_nodes / initial_num_nodes if initial_num_nodes > 0 else 0
                        percent_edges_retained = num_edges / initial_num_edges if initial_num_edges > 0 else 0

                        # Compute graph metrics
                        try:
                            sub_communities = nx_comm.louvain_communities(graph, seed=42)
                            modularity_score = modularity(graph, sub_communities)
                            num_communities = len(sub_communities)
                            avg_community_size = sum(len(c) for c in sub_communities) / num_communities if num_communities > 0 else 0
                            avg_clustering = nx.average_clustering(graph)
                            graph_density = nx.density(graph)
                            avg_degree = sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
                        except Exception as e:
                            logger.error(f"Failed to compute graph metrics: {e}")
                            modularity_score = num_communities = avg_community_size = avg_clustering = graph_density = avg_degree = 0

                        # Check for isolated test nodes
                        isolated_test_nodes = {n for n in fixed_test_nodes if n in graph_nodes and graph.degree(n) == 0}
                        if isolated_test_nodes:
                            logger.warning(f"{len(isolated_test_nodes)} isolated test nodes detected")

                        # Prepare node index mapping for GNN
                        graph_nodes_list = list(graph_nodes)
                        node_to_index = {node: idx for idx, node in enumerate(graph_nodes_list)}
                        train_indices = [node_to_index[node] for node in graph_nodes_list if node in graph_nodes and node not in fixed_test_nodes]
                        test_indices = [node_to_index[node] for node in graph_nodes_list if node in fixed_test_nodes and node in graph_nodes]

                        # Prepare edge indices with reindexing
                        try:
                            edge_index = torch.tensor(
                                [(node_to_index[src], node_to_index[dst]) for src, dst in graph.edges
                                 if src in node_to_index and dst in node_to_index],
                                dtype=torch.long
                            ).t().contiguous()
                            weights = np.array([d['weight'] for _, _, d in graph.edges(data=True)])
                            if len(weights) > 0 and (weights.min() < 0 or weights.max() > 1):
                                logger.info(f"Normalizing edge weights: min={weights.min():.4f}, max={weights.max():.4f}")
                                weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
                            edge_weight = torch.tensor(weights, dtype=torch.float) if use_edge_weights else None
                        except Exception as e:
                            logger.error(f"Failed to process edge indices or weights: {e}")
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_weight = None

                        # Prepare GNN data
                        graph_indices = [summary_data.index.get_loc(node) for node in graph_nodes_list if node in summary_data.index]
                        if not graph_indices:
                            logger.error(f"No valid nodes for GNN data. Skipping method {method}, threshold {threshold}, run {run}.")
                            pbar.update(len(models))
                            continue

                        x_subset = x[graph_indices]
                        y_subset = y[graph_indices]
                        train_mask_subset = torch.tensor([node_to_index[node] for node in graph_nodes_list
                                                        if node in node_to_index and node not in fixed_test_nodes], dtype=torch.long)
                        test_mask_subset = torch.tensor([node_to_index[node] for node in graph_nodes_list
                                                       if node in node_to_index and node in fixed_test_nodes], dtype=torch.long)
                        logger.info(f"GNN data: {len(graph_nodes_list)} nodes, {x_subset.shape[0]} features, train_mask={len(train_mask_subset)}, test_mask={len(test_mask_subset)}")

                        # Validate data consistency
                        if x_subset.shape[0] != len(graph_nodes_list):
                            logger.error(f"Mismatch: feature matrix has {x_subset.shape[0]} nodes, graph has {len(graph_nodes_list)} nodes")
                            pbar.update(len(models))
                            continue
                        if edge_index.shape[1] > 0 and edge_index.max().item() >= len(graph_nodes_list):
                            logger.error(f"Edge index contains invalid node indices: max={edge_index.max().item()}, num_nodes={len(graph_nodes_list)}")
                            pbar.update(len(models))
                            continue

                        # Create data object
                        data = type('Data', (), {})()
                        data.x = x_subset
                        data.edge_index = edge_index
                        data.edge_weight = edge_weight
                        data.y = y_subset
                        data.train_mask = train_mask_subset
                        data.test_mask = test_mask_subset

                        for model_name in models:
                            experiment_counter += 1
                            experiment_id = (threshold, method, run, model_name)
                            progress = f"Experiment {experiment_counter}/{total_experiments} (threshold={threshold:.2f}, method={method}, run={run}, model={model_name})"
                            logger.info(f"Starting {progress}")

                            if experiment_id in completed_evaluation:
                                logger.info(f"Skipping completed experiment: {experiment_id}")
                                pbar.update(1)
                                continue

                            result = {
                                "Threshold": threshold,
                                "Method": method,
                                "Run": run,
                                "Model": model_name,
                                "Num_Edges": num_edges,
                                "Num_Nodes": num_nodes,
                                "Modularity": modularity_score,
                                "Num_Communities": num_communities,
                                "Avg_Community_Size": avg_community_size,
                                "Avg_Clustering": avg_clustering,
                                "Graph_Density": graph_density,
                                "Avg_Degree": avg_degree,
                                "Nodes_Removed": nodes_removed,
                                "Edges_Removed": edges_removed,
                                "Percent_Nodes_Retained": percent_nodes_retained,
                                "Percent_Edges_Retained": percent_edges_retained,
                                "Test_Set_Size": len(test_mask_subset),
                                "Isolated_Test_Nodes": len(isolated_test_nodes)
                            }
                            if task == 'classification':
                                result.update({
                                    "Accuracy": 0.0,
                                    "Confusion_Matrix": [],
                                    "Cohen_Kappa": 0.0
                                })
                            else:  # regression
                                result.update({
                                    "MAE": 0.0,
                                    "RMSE": 0.0,
                                    "R2": 0.0
                                })

                            try:
                                logger.info(f"Training {model_name} for {progress}")
                                start_training_time = time.time()
                                model = {
                                    "GAT": GAT(
                                        dim_in=x.shape[1],
                                        dim_h=64,
                                        dim_out=1 if task == 'regression' else len(np.unique(labels))
                                    ),
                                    "GCN": GCN(
                                        dim_in=x.shape[1],
                                        dim_h=64,
                                        dim_out=1 if task == 'regression' else len(np.unique(labels))
                                    ),
                                    "GraphSAGE": GraphSAGE(
                                        dim_in=x.shape[1],
                                        dim_h=64,
                                        dim_out=1 if task == 'regression' else len(np.unique(labels))
                                    )
                                }[model_name]
                                model = train(model, data, class_weights=class_weights if task == 'classification' else None, task=task)
                                training_time = time.time() - start_training_time
                                peak_memory = max(peak_memory, log_memory_usage())
                                performance_metrics.append({
                                    'Threshold': threshold,
                                    'Method': method,
                                    'Run': run,
                                    'Model': model_name,
                                    'Training_Time': training_time,
                                    'Peak_Memory_MB': peak_memory
                                })
                                pd.DataFrame(performance_metrics).to_csv(performance_metrics_file, index=False)
                                logger.info(f"Updated performance metrics to {performance_metrics_file}")
                                metrics = test(model, data, task=task)
                                if task == 'classification':
                                    acc, cm, kappa = metrics
                                    test_set_size = sum(sum(row) for row in cm)
                                    result.update({
                                        "Accuracy": acc,
                                        "Confusion_Matrix": cm.tolist(),
                                        "Cohen_Kappa": kappa,
                                        "Test_Set_Size": test_set_size
                                    })
                                    logger.info(f"Completed {progress}: Accuracy={acc:.4f}, Cohen_Kappa={kappa:.4f}, Test_Set_Size={test_set_size}")
                                else:  # regression
                                    mae, rmse, r2 = metrics
                                    result.update({
                                        "MAE": mae,
                                        "RMSE": rmse,
                                        "R2": r2
                                    })
                                    logger.info(f"Completed {progress}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Test_Set_Size={len(test_mask_subset)}")
                            except Exception as e:
                                logger.error(f"Error in experiment for {progress}: {str(e)}\n{traceback.format_exc()}")
                                results.append(result)
                                save_results(results, threshold, method, run, single_file=True, task=task)
                                pbar.update(1)
                                continue

                            results = [r for r in results if not (
                                r['Threshold'] == threshold and
                                r['Method'] == method and
                                r['Run'] == run and
                                r['Model'] == model_name
                            )]
                            results.append(result)
                            save_results(results, threshold, method, run, single_file=True, task=task)
                            completed_evaluation.add(experiment_id)
                            checkpoint = {
                                "completed_sparsification": list(completed_sparsification),
                                "completed_evaluation": list(completed_evaluation),
                                "results": results,
                                "fixed_test_nodes": list(fixed_test_nodes)
                            }
                            save_checkpoint(checkpoint_file, checkpoint)
                            pbar.update(1)
            except Exception as e:
                logger.error(f"Error at threshold {threshold}: {str(e)}\n{traceback.format_exc()}")
                continue

    # Aggregate results
    try:
        if results:
            unique_results = []
            seen = set()
            for r in results:
                key = (r['Threshold'], r['Method'], r['Run'], r['Model'])
                if key not in seen:
                    unique_results.append(r)
                    seen.add(key)
            aggregate_results(unique_results, num_runs=1, task=task)
            logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        else:
            logger.error("No results were generated")
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}\n{traceback.format_exc()}")
        save_checkpoint(checkpoint_file, {
            "completed_sparsification": list(completed_sparsification),
            "completed_evaluation": list(completed_evaluation),
            "results": results,
            "fixed_test_nodes": list(fixed_test_nodes)
        })