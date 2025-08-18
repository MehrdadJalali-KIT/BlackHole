import pandas as pd
import numpy as np
import networkx as nx
import logging
import os
from itertools import combinations

logging.basicConfig(level=logging.INFO, filename="bh_evaluation.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def sparsify_edges(edges_list, target_num_nodes, method, valid_nodes, target_num_edges, fixed_test_nodes, pagerank_scores=None):
    """Sparsifies edges to match target_num_nodes and target_num_edges, ensuring fixed test nodes are retained."""
    if target_num_nodes == 0 or target_num_edges == 0:
        logger.warning(f"No nodes or edges to keep for method {method}")
        return pd.DataFrame(columns=edges_list.columns)

    # Ensure fixed test nodes are included
    num_fixed_nodes = len(fixed_test_nodes)
    target_non_test_nodes = max(0, target_num_nodes - num_fixed_nodes)
    logger.info(f"Method {method}: Fixed test nodes: {num_fixed_nodes}, Target non-test nodes: {target_non_test_nodes}")

    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(fixed_test_nodes)  # Always include fixed test nodes
    valid_non_test_nodes = [node for node in valid_nodes if node not in fixed_test_nodes]
    selected_edges = []
    selected_nodes = set(fixed_test_nodes)

    if method == "blackhole":
        return edges_list

    elif method == "random":
        # Shuffle non-test nodes and select target_non_test_nodes
        np.random.seed(42)
        if len(valid_non_test_nodes) > target_non_test_nodes:
            selected_non_test_nodes = set(np.random.choice(valid_non_test_nodes, size=target_non_test_nodes, replace=False))
        else:
            selected_non_test_nodes = set(valid_non_test_nodes)
        selected_nodes.update(selected_non_test_nodes)
        
        # Filter edges to only include selected nodes
        candidate_edges = edges_list[
            (edges_list['source'].isin(selected_nodes)) & (edges_list['target'].isin(selected_nodes))
        ].sample(frac=1, random_state=42).reset_index(drop=True)
        
        for _, edge in candidate_edges.iterrows():
            src, tgt = edge['source'], edge['target']
            temp_graph.add_edge(src, tgt, weight=edge['weight'])
            selected_edges.append(edge)
            if len(selected_edges) >= target_num_edges:
                break
        
        # If fewer edges, add edges between selected nodes
        if len(selected_edges) < target_num_edges:
            remaining_nodes = list(selected_nodes)
            possible_edges = [(src, tgt) for src, tgt in combinations(remaining_nodes, 2) if not temp_graph.has_edge(src, tgt)]
            np.random.shuffle(possible_edges)
            for src, tgt in possible_edges:
                if len(selected_edges) >= target_num_edges:
                    break
                weight = edges_list['weight'].mean()
                temp_graph.add_edge(src, tgt, weight=weight)
                selected_edges.append(pd.Series({'source': src, 'target': tgt, 'weight': weight}))
        
        # Trim to exact number of edges
        if len(selected_edges) > target_num_edges:
            selected_edges = selected_edges[:target_num_edges]
        
        # Rebuild graph to ensure node count
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(selected_nodes)
        for edge in selected_edges:
            temp_graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        final_nodes = set(temp_graph.nodes())
        if len(final_nodes) != target_num_nodes:
            logger.warning(f"Random method selected {len(final_nodes)} nodes (target: {target_num_nodes})")
        
        return pd.DataFrame(selected_edges, columns=edges_list.columns)

    elif method == "stratified":
        # Select non-test nodes based on degree to match target_non_test_nodes
        temp_graph = nx.from_pandas_edgelist(edges_list, 'source', 'target', 'weight')
        valid_non_test_nodes = [node for node in valid_nodes if node not in fixed_test_nodes]
        degrees = [(node, temp_graph.degree(node)) for node in valid_non_test_nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        selected_non_test_nodes = set(node for node, _ in degrees[:target_non_test_nodes])
        selected_nodes.update(selected_non_test_nodes)
        
        # Bin edges by weight
        bins = np.histogram(edges_list['weight'], bins=10)[1]
        edges_per_bin = max(1, target_num_edges // (len(bins) - 1))
        selected_edges = []
        
        for i in range(len(bins) - 1):
            bin_mask = (edges_list['weight'] >= bins[i]) & (edges_list['weight'] < bins[i + 1])
            bin_edges = edges_list[bin_mask & edges_list['source'].isin(selected_nodes) & edges_list['target'].isin(selected_nodes)]
            bin_edges = bin_edges.sample(frac=1, random_state=42).reset_index(drop=True)
            edges_added = 0
            for _, edge in bin_edges.iterrows():
                if edges_added >= edges_per_bin or len(selected_edges) >= target_num_edges:
                    break
                temp_graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
                selected_edges.append(edge)
                edges_added += 1
        
        # Fill remaining edges
        if len(selected_edges) < target_num_edges:
            remaining_edges = edges_list[
                (edges_list['source'].isin(selected_nodes)) & (edges_list['target'].isin(selected_nodes)) &
                (~edges_list.index.isin([e.name for e in selected_edges]))
            ].sample(frac=1, random_state=42).reset_index(drop=True)
            for _, edge in remaining_edges.iterrows():
                if len(selected_edges) >= target_num_edges:
                    break
                temp_graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
                selected_edges.append(edge)
        
        # If still short, add edges
        if len(selected_edges) < target_num_edges:
            remaining_nodes = list(selected_nodes)
            possible_edges = [(src, tgt) for src, tgt in combinations(remaining_nodes, 2) if not temp_graph.has_edge(src, tgt)]
            np.random.shuffle(possible_edges)
            for src, tgt in possible_edges:
                if len(selected_edges) >= target_num_edges:
                    break
                weight = edges_list['weight'].mean()
                temp_graph.add_edge(src, tgt, weight=weight)
                selected_edges.append(pd.Series({'source': src, 'target': tgt, 'weight': weight}))
        
        # Trim to exact number of edges
        if len(selected_edges) > target_num_edges:
            selected_edges = selected_edges[:target_num_edges]
        
        # Rebuild graph
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(selected_nodes)
        for edge in selected_edges:
            temp_graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        final_nodes = set(temp_graph.nodes())
        if len(final_nodes) != target_num_nodes:
            logger.warning(f"Stratified method selected {len(final_nodes)} nodes (target: {target_num_nodes})")
        
        return pd.DataFrame(selected_edges, columns=edges_list.columns)

    elif method == "pagerank":
        if pagerank_scores is None:
            raise ValueError("PageRank scores must be provided for pagerank method")
        
        # Select top non-test nodes by PageRank
        valid_non_test_nodes = [node for node in valid_nodes if node not in fixed_test_nodes]
        pagerank_list = [(node, pagerank_scores.get(node, 0)) for node in valid_non_test_nodes]
        pagerank_list.sort(key=lambda x: x[1], reverse=True)
        selected_non_test_nodes = set(node for node, _ in pagerank_list[:target_non_test_nodes])
        selected_nodes.update(selected_non_test_nodes)
        
        # Filter edges to only include selected nodes (induced subgraph)
        candidate_edges = edges_list[
            (edges_list['source'].isin(selected_nodes)) & (edges_list['target'].isin(selected_nodes))
        ].sample(frac=1, random_state=42).reset_index(drop=True)
        
        for _, edge in candidate_edges.iterrows():
            src, tgt = edge['source'], edge['target']
            temp_graph.add_edge(src, tgt, weight=edge['weight'])
            selected_edges.append(edge)
            if len(selected_edges) >= target_num_edges:
                break
        
        # If fewer edges, add edges between selected nodes
        if len(selected_edges) < target_num_edges:
            remaining_nodes = list(selected_nodes)
            possible_edges = [(src, tgt) for src, tgt in combinations(remaining_nodes, 2) if not temp_graph.has_edge(src, tgt)]
            np.random.shuffle(possible_edges)
            for src, tgt in possible_edges:
                if len(selected_edges) >= target_num_edges:
                    break
                weight = edges_list['weight'].mean()
                temp_graph.add_edge(src, tgt, weight=weight)
                selected_edges.append(pd.Series({'source': src, 'target': tgt, 'weight': weight}))
        
        # Trim to exact number of edges
        if len(selected_edges) > target_num_edges:
            selected_edges = selected_edges[:target_num_edges]
        
        # Rebuild graph
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(selected_nodes)
        for edge in selected_edges:
            temp_graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        final_nodes = set(temp_graph.nodes())
        if len(final_nodes) != target_num_nodes:
            logger.warning(f"PageRank method selected {len(final_nodes)} nodes (target: {target_num_nodes})")
        
        return pd.DataFrame(selected_edges, columns=edges_list.columns)

    elif method == "kcenter":
        import random
        G = nx.from_pandas_edgelist(edges_list, 'source', 'target', edge_attr=True)

        # How many non-test nodes to select in addition to fixed test nodes
        k = int(target_non_test_nodes) if 'target_non_test_nodes' in locals() else 0

        # Candidate pool: valid non-test nodes (ensure they exist in the graph)
        candidates = [u for u in valid_non_test_nodes if u in G]

        # Selected set always includes fixed test nodes
        selected = set([u for u in fixed_test_nodes if u in G])

        # Helper: update min distances with a single-source BFS
        INF = 10**9
        min_dist = {u: (0 if u in selected else INF) for u in candidates}

        # Seed center: pick a high-degree candidate if available (improves coverage early)
        if candidates and (len(selected) == 0):
            seed = max(candidates, key=lambda u: G.degree(u))
            selected.add(seed)
            lengths = nx.single_source_shortest_path_length(G, seed)
            for u in candidates:
                d = lengths.get(u, INF)
                if d < min_dist[u]:
                    min_dist[u] = d
            if seed in candidates:
                candidates.remove(seed)

        # Greedy farthest-first: iteratively add the candidate with the largest min distance
        # until we reach k selected non-test nodes (or run out of candidates)
        while len(selected.difference(set(fixed_test_nodes))) < k and candidates:
            # Choose the farthest candidate under current min distances (break ties by degree)
            v = max(candidates, key=lambda u: (min_dist[u], G.degree(u)))
            selected.add(v)

            # Update min distances using a single BFS from v
            lengths = nx.single_source_shortest_path_length(G, v)
            for u in candidates:
                d = lengths.get(u, INF)
                if d < min_dist[u]:
                    min_dist[u] = d

            # Remove v from candidates
            try:
                candidates.remove(v)
            except ValueError:
                pass

        # Final node set: fixed test nodes + up to k non-test k-center nodes
        selected_nodes = selected

        # Induce subgraph edges over selected nodes
        mask = edges_list['source'].isin(selected_nodes) & edges_list['target'].isin(selected_nodes)
        sparse_edges = edges_list.loc[mask].copy()

        # If needed, trim to target_num_edges while keeping higher-weight edges first when available
        try:
            if target_num_edges and len(sparse_edges) > target_num_edges:
                if 'weight' in sparse_edges.columns:
                    sparse_edges = sparse_edges.sort_values('weight', ascending=False).head(int(target_num_edges))
            else:
                # fallback: random sample (deterministic if run uses a seed elsewhere)
                sparse_edges = sparse_edges.sample(n=int(target_num_edges), random_state=42)
        except Exception as _e:
            # If trimming fails, keep current sparse_edges as-is
            pass

        # Ensure columns align with the input schema
        sparse_edges = sparse_edges.reset_index(drop=True)
        return sparse_edges.reset_index(drop=True)
        
        
    else:
        raise ValueError(f"Unknown sparsification method: {method}")


def save_sparsified_edges(sparse_edges, threshold, method, run):
    """Saves sparsified edges for non-BH methods."""
    output_dir = f"sparsified_graphs/threshold_{threshold:.2f}/method_{method}/run_{run}"
    os.makedirs(output_dir, exist_ok=True)
    sparse_edges.to_csv(os.path.join(output_dir, f"edges_t{threshold:.2f}_r{run}.csv"), index=False)
    logger.info(f"Saved {method} edges to {output_dir}/edges_t{threshold:.2f}_r{run}.csv")