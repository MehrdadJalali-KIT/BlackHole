import pandas as pd
import numpy as np
import networkx as nx
import logging
from sklearn.preprocessing import MinMaxScaler
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="bh_evaluation.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def calculate_gravity_per_community(graph, communities, weights=(0.3, 0.3, 0.4)):
    """
    Calculate normalized gravity scores and identify top nodes per community.
    Returns gravity scores, centralities, edge weight sums, and test nodes with degree > 2.
    """
    community_gravity = {}
    degree_centrality = {}
    betweenness_centrality = {}
    edge_weight_sum = {}
    test_nodes = set()
    degree_weight, betweenness_weight, weight_sum_weight = weights

    for idx, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        if subgraph.number_of_nodes() == 0:
            logger.warning(f"Empty community {idx}, skipping")
            continue

        # Compute metrics
        degree_centrality_community = nx.degree_centrality(subgraph)
        betweenness_centrality_community = nx.betweenness_centrality(subgraph, normalized=True)
        edge_weight_sum_community = {
            node: sum(data['weight'] for _, _, data in subgraph.edges(node, data=True))
            for node in subgraph.nodes()
        }

        # Normalize metrics
        degree_values = list(degree_centrality_community.values())
        betweenness_values = list(betweenness_centrality_community.values())
        weight_sum_values = list(edge_weight_sum_community.values())

        scaler = MinMaxScaler()
        normalized_degree = scaler.fit_transform(np.array(degree_values).reshape(-1, 1)).flatten()
        normalized_betweenness = scaler.fit_transform(np.array(betweenness_values).reshape(-1, 1)).flatten()
        normalized_weight_sum = scaler.fit_transform(np.array(weight_sum_values).reshape(-1, 1)).flatten()

        # Assign gravity and metrics
        for idx, node in enumerate(community):
            gravity = (
                degree_weight * normalized_degree[idx] +
                betweenness_weight * normalized_betweenness[idx] +
                weight_sum_weight * normalized_weight_sum[idx]
            )
            community_gravity[node] = gravity
            degree_centrality[node] = degree_centrality_community[node]
            betweenness_centrality[node] = betweenness_centrality_community[node]
            edge_weight_sum[node] = edge_weight_sum_community[node]

        # Select top 4% nodes with degree > 2 per community as test nodes
        node_degrees = {node: subgraph.degree(node) for node in community}
        eligible_nodes = [node for node, degree in node_degrees.items() if degree > 2]
        if not eligible_nodes:
            logger.warning(f"Community {idx} has no nodes with degree > 2, selecting highest-degree node")
            eligible_nodes = [max(node_degrees.items(), key=lambda x: x[1])[0]] if node_degrees else []
        if eligible_nodes:
            community_gravity_scores = [(node, community_gravity[node]) for node in eligible_nodes]
            community_gravity_scores.sort(key=lambda x: x[1], reverse=True)
            num_test_nodes = max(1, int(0.04 * len(community)))  # At least 1 node
            selected_test_nodes = [node for node, _ in community_gravity_scores[:num_test_nodes]]
            test_nodes.update(selected_test_nodes)
            logger.info(f"Community {idx} size {len(community)}: selected {len(selected_test_nodes)} test nodes with degree > 2")

    return community_gravity, degree_centrality, betweenness_centrality, edge_weight_sum, test_nodes

def black_hole_strategy_per_community(graph, gravity, communities, threshold, fixed_test_nodes, summary_data):
    """
    Retain top (1 - threshold) nodes by gravity in each community, stratified by PLD distribution.
    Returns the graph and number of retained nodes.
    """
    nodes_to_remove = []
    nodes_retained = 0
    total_nodes = sum(len(community) for community in communities)
    selected_nodes = set(fixed_test_nodes)

    if total_nodes == 0:
        logger.warning("No nodes in communities, returning empty graph")
        return graph, 0

    # Calculate target number of nodes to retain (excluding fixed test nodes)
    num_fixed_nodes = len(fixed_test_nodes)
    target_num_nodes = int(max(0.2 * graph.number_of_nodes(), (1 - threshold) * graph.number_of_nodes()))
    target_non_test_nodes = max(0, target_num_nodes - num_fixed_nodes)
    logger.info(f"Fixed test nodes: {num_fixed_nodes}, Target non-test nodes: {target_non_test_nodes}")

    # Distribute target_non_test_nodes proportionally across communities
    community_sizes = [len(community) for community in communities]
    community_targets = [
        max(1, int(target_non_test_nodes * (size / total_nodes))) for size in community_sizes
    ]
    total_assigned = sum(community_targets)
    if total_assigned != target_non_test_nodes:
        diff = target_non_test_nodes - total_assigned
        for i in range(abs(diff)):
            idx = i % len(community_targets)
            community_targets[idx] += 1 if diff > 0 else -1
            community_targets[idx] = max(0, community_targets[idx])

    # Stratify by PLD category within each community
    for idx, (community, nodes_to_keep) in enumerate(zip(communities, community_targets)):
        community_nodes = [node for node in community if node in gravity and node not in fixed_test_nodes]
        community_test_nodes = [node for node in community if node in fixed_test_nodes]
        if not community_nodes and not community_test_nodes:
            logger.warning(f"Community {idx} has no nodes with gravity scores, skipping")
            continue

        # Get PLD categories for community nodes
        community_summary = summary_data[summary_data.index.isin(community_nodes)]
        if community_summary.empty:
            logger.warning(f"Community {idx} has no nodes in summary_data, selecting by gravity only")
            community_gravity_scores = [(node, gravity[node]) for node in community_nodes]
            community_gravity_scores.sort(key=lambda x: x[1], reverse=True)
            nodes_to_keep = min(nodes_to_keep, len(community_gravity_scores))
            selected_nodes.update(node for node, _ in community_gravity_scores[:nodes_to_keep])
            nodes_to_remove.extend(node for node, _ in community_gravity_scores[nodes_to_keep:])
            logger.info(f"Community {idx} size {len(community_nodes)}: kept {len(community_test_nodes)} test nodes, {nodes_to_keep} non-test nodes")
            continue

        # Compute PLD distribution
        pld_counts = community_summary['category'].value_counts(normalize=True).to_dict()
        nodes_to_keep_per_category = {
            cat: max(1, int(nodes_to_keep * proportion)) for cat, proportion in pld_counts.items()
        }
        total_assigned_category = sum(nodes_to_keep_per_category.values())
        if total_assigned_category != nodes_to_keep:
            diff = nodes_to_keep - total_assigned_category
            for cat in nodes_to_keep_per_category:
                nodes_to_keep_per_category[cat] += diff // len(nodes_to_keep_per_category)
                diff -= diff // len(nodes_to_keep_per_category)

        # Select nodes per PLD category
        for category, target_count in nodes_to_keep_per_category.items():
            category_nodes = community_summary[community_summary['category'] == category].index.tolist()
            category_gravity_scores = [(node, gravity[node]) for node in category_nodes if node in gravity]
            category_gravity_scores.sort(key=lambda x: x[1], reverse=True)
            selected_category_nodes = [node for node, _ in category_gravity_scores[:target_count]]
            selected_nodes.update(selected_category_nodes)
            logger.info(f"Community {idx}, category {category}: selected {len(selected_category_nodes)}/{target_count} nodes")

        # Remove unselected nodes
        unselected_nodes = [node for node in community_nodes if node not in selected_nodes]
        nodes_to_remove.extend(unselected_nodes)
        nodes_retained += len(community_test_nodes) + len(selected_nodes.intersection(community_nodes))
        logger.info(f"Community {idx} size {len(community_nodes) + len(community_test_nodes)}: kept {len(community_test_nodes)} test nodes, {len(selected_nodes.intersection(community_nodes))} non-test nodes")

    # Remove nodes and update graph
    graph.remove_nodes_from(nodes_to_remove)
    nodes_retained = graph.number_of_nodes()
    logger.info(f"Retained {nodes_retained} nodes after BH strategy (including {num_fixed_nodes} fixed test nodes)")
    return graph, nodes_retained

def prune_edges(graph, edge_threshold, fixed_test_nodes):
    """Prune edges below the top (1 - edge_threshold) fraction of weights, ensuring each test node has at least one edge if possible."""
    edges = [(u, v, d['weight']) for u, v, d in graph.edges(data=True)]
    if not edges:
        logger.warning("No edges to prune")
        return graph

    # Identify edges connected to test nodes
    test_node_edges = [(u, v, w) for u, v, w in edges if u in fixed_test_nodes or v in fixed_test_nodes]
    non_test_edges = [(u, v, w) for u, v, w in edges if u not in fixed_test_nodes and v not in fixed_test_nodes]

    # Ensure each test node has at least one edge (highest weight if available)
    test_nodes_with_edges = set()
    test_edges_to_keep = []
    for test_node in fixed_test_nodes:
        if test_node not in graph:
            continue
        node_edges = [(u, v, w) for u, v, w in test_node_edges if u == test_node or v == test_node]
        if node_edges:
            # Select the highest-weight edge for this test node
            highest_edge = max(node_edges, key=lambda x: x[2])
            test_edges_to_keep.append(highest_edge)
            test_nodes_with_edges.add(test_node)

    # Log test nodes without edges
    test_nodes_without_edges = fixed_test_nodes - test_nodes_with_edges
    if test_nodes_without_edges:
        logger.warning(f"{len(test_nodes_without_edges)} test nodes have no edges: {list(test_nodes_without_edges)[:10]}")

    # Prune remaining edges
    num_edges_to_keep = max(10, int((1 - edge_threshold) * len(edges))) - len(test_edges_to_keep)
    num_edges_to_keep = max(0, num_edges_to_keep)  # Ensure non-negative
    non_test_edges.sort(key=lambda x: x[2], reverse=True)
    edges_to_keep = test_edges_to_keep + non_test_edges[:num_edges_to_keep]

    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes())
    new_graph.add_weighted_edges_from(edges_to_keep)
    logger.info(f"Pruned to {len(edges_to_keep)} edges, ensured {len(test_nodes_with_edges)} test nodes have at least one edge")
    return new_graph

def save_extended_node_features(graph, original_summary_data, gravity, degree_centrality, betweenness_centrality, edge_weight_sum, communities, filename):
    """Save node features, including gravity and community IDs, for remaining nodes."""
    remaining_nodes = list(graph.nodes())
    try:
        filtered_summary_data = original_summary_data.loc[
            original_summary_data.index.intersection(remaining_nodes)
        ]

        community_map = {
            node: idx for idx, community in enumerate(communities)
            for node in community if node in remaining_nodes
        }

        metrics_data = pd.DataFrame({
            'refcode': remaining_nodes,
            'Gravity': [gravity.get(node, 0) for node in remaining_nodes],
            'Degree_Centrality': [degree_centrality.get(node, 0) for node in remaining_nodes],
            'Betweenness_Centrality': [betweenness_centrality.get(node, 0) for node in remaining_nodes],
            'Edge_Weight_Sum': [edge_weight_sum.get(node, 0) for node in remaining_nodes],
            'Community_ID': [community_map.get(node, -1) for node in remaining_nodes]
        })
        metrics_data.set_index('refcode', inplace=True)

        final_data = pd.concat([filtered_summary_data, metrics_data], axis=1)
        final_data.to_csv(filename, index=True)
        logger.info(f"Saved node features to {filename}")
    except Exception as e:
        logger.error(f"Failed to save node features to {filename}: {e}")

def generate_bootstrapped_blackhole_edges(edges_list, n_bootstrap_edges, threshold, run, communities, gravity, degree_centrality, betweenness_centrality, edge_weight_sum, test_nodes, summary_data):
    """
    Generate bootstrapped BH sparsified edges using precomputed communities and gravity scores.
    """
    try:
        # Bootstrap sample edges
        sampled_edges = edges_list.sample(n=n_bootstrap_edges, replace=True, random_state=run)
        graph = nx.Graph()
        for _, row in sampled_edges.iterrows():
            graph.add_edge(row['source'], row['target'], weight=row['weight'])

        logger.info(f"Bootstrapped graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            logger.warning(f"Empty graph for threshold {threshold}, run {run}")
            return pd.DataFrame(columns=['source', 'target', 'weight']), 0

        # Apply black hole strategy with stratification
        graph, nodes_retained = black_hole_strategy_per_community(graph, gravity, communities, threshold, test_nodes, summary_data)

        # Prune edges, ensuring test node connectivity
        graph = prune_edges(graph, threshold, test_nodes)

        # Save node features
        output_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_blackhole/run_{run}"
        os.makedirs(output_dir, exist_ok=True)
        output_features_filename = os.path.join(output_dir, f"remaining_node_features_t{threshold:.2f}_r{run}.csv")
        save_extended_node_features(
            graph,
            summary_data,
            gravity,
            degree_centrality,
            betweenness_centrality,
            edge_weight_sum,
            communities,
            output_features_filename
        )

        # Save edges
        edges = [(u, v, d['weight']) for u, v, d in graph.edges(data=True)]
        bh_edges = pd.DataFrame(edges, columns=['source', 'target', 'weight'])

        output_edge_filename = os.path.join(output_dir, f"BH_edges_t{threshold:.2f}_r{run}.csv")
        bh_edges.to_csv(output_edge_filename, index=False)
        logger.info(f"Saved BH edges to {output_edge_filename}, {len(bh_edges)} edges, {nodes_retained} nodes")

        return bh_edges, nodes_retained

    except Exception as e:
        logger.error(f"Error in BH sparsification for threshold {threshold}, run {run}: {e}")
        return pd.DataFrame(columns=['source', 'target', 'weight']), 0