import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, save_npz
import multiprocessing as mp
import networkx as nx
import torch
from collections import defaultdict

def convert_multidigraph_to_graph(multi_g):
    """
    Convert a MultiDiGraph to an undirected Graph by collapsing multiple edges into single edges.
    """
    graph = nx.Graph()
    graph.add_nodes_from(multi_g.nodes(data=True))
    for u, v, data in multi_g.edges(data=True):
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, **data)
    return graph


def compute_m_u(graph, alpha=0.5, p=0):
    """
    Compute probability measure m_u for all nodes using uniform measure.
    """
    m_u_dict = {}
    for u in graph.nodes():
        neighbors = list(graph.neighbors(u))
        degree_u = graph.degree(u, weight='weight')  # Get weighted degree
        m_u = {u: alpha}
        if degree_u > 0:
            m_u.update({v: (1 - alpha) / degree_u for v in neighbors})
        m_u_dict[u] = m_u
    return m_u_dict


import networkx as nx
import torch
from collections import defaultdict

import networkx as nx
import torch
from collections import defaultdict

import networkx as nx
import torch
from collections import defaultdict

def compute_orc_bounds(rank, graph, edge_batch, result_queue, device):
    """
    Compute the exact upper and lower bounds of Ollivier-Ricci curvature (ORC) for a weighted graph.

    Args:
        rank: the GPU id for distributed computation.
        graph: A networkx graph with edge weights.
        edge_batch: Batch of edges assigned to this worker.
        result_queue: The queue to gather results from all processes.
        device: The device to use (e.g., 'cuda:0').
    """
    upper_bounds = {}
    lower_bounds = {}

    for edge in edge_batch:
        u, v = edge[:2]
        w_uv = graph[u][v]['weight']

        # Compute probability measures m_u and m_v
        m_u = defaultdict(float)
        m_v = defaultdict(float)

        # For uniform measure with alpha=0.5
        alpha = 0.5

        # Compute degrees (or weighted degrees)
        deg_u = sum(graph[u][nbr].get('weight', 1) for nbr in graph.neighbors(u))
        deg_v = sum(graph[v][nbr].get('weight', 1) for nbr in graph.neighbors(v))

        # Normalization constants
        C_u = deg_u
        C_v = deg_v

        # Measure for u
        m_u[u] = alpha
        for nbr in graph.neighbors(u):
            m_u[nbr] = (1 - alpha) * graph[u][nbr].get('weight', 1) / C_u

        # Measure for v
        m_v[v] = alpha
        for nbr in graph.neighbors(v):
            m_v[nbr] = (1 - alpha) * graph[v][nbr].get('weight', 1) / C_v

        # Nodes to consider
        nodes = set(m_u.keys()).union(set(m_v.keys()))

        # Compute delta_m
        delta_m = {}
        for node in nodes:
            delta_m[node] = m_u.get(node, 0) - m_v.get(node, 0)

        # Identify sets P and N
        P = {node for node in nodes if delta_m[node] > 0}
        N = {node for node in nodes if delta_m[node] < 0}

        # Initialize distances
        dist_to_N = {}
        dist_to_P = {}

        # Compute distances only if N or P is not empty
        if N:
            # Precompute shortest path lengths from each node to any node in N
            for node in nodes:
                if node in N:
                    dist_to_N[node] = 0
                else:
                    try:
                        # Compute distances to all nodes in N
                        distances = (nx.shortest_path_length(graph, source=node, target=n) for n in N)
                        dist_N = min(distances)
                        dist_to_N[node] = dist_N
                    except (nx.NetworkXNoPath, ValueError):
                        dist_to_N[node] = float('inf')
        else:
            # If N is empty, set distances to infinity
            for node in nodes:
                dist_to_N[node] = float('inf')

        if P:
            # Precompute shortest path lengths from each node to any node in P
            for node in nodes:
                if node in P:
                    dist_to_P[node] = 0
                else:
                    try:
                        # Compute distances to all nodes in P
                        distances = (nx.shortest_path_length(graph, source=node, target=p) for p in P)
                        dist_P = min(distances)
                        dist_to_P[node] = dist_P
                    except (nx.NetworkXNoPath, ValueError):
                        dist_to_P[node] = float('inf')
        else:
            # If P is empty, set distances to infinity
            for node in nodes:
                dist_to_P[node] = float('inf')

        # Upper bound computation
        sum_P = sum(
            dist_to_N[node] * delta_m[node]
            for node in nodes
            if delta_m[node] > 0 and dist_to_N[node] != float('inf')
        )

        sum_N = sum(
            dist_to_P[node] * (-delta_m[node])
            for node in nodes
            if delta_m[node] < 0 and dist_to_P[node] != float('inf')
        )

        # Handle cases where sums are zero (to avoid division by zero)
        if sum_P == 0 and sum_N == 0:
            upper_bound = 1
        else:
            upper_bound = 1 - (1 / w_uv) * max(sum_P, sum_N)

        upper_bounds[(u, v)] = upper_bound

        # Lower bound computation remains the same
        sum_l = sum((graph[u][l]['weight'] / w_uv) * m_u[l] for l in graph.neighbors(u))
        sum_r = sum((graph[v][r]['weight'] / w_uv) * m_v[r] for r in graph.neighbors(v))

        # Compute the sum over common neighbors
        common_neighbors = set(graph.neighbors(u)).intersection(set(graph.neighbors(v)))
        sum_c = 0
        for c in common_neighbors:
            delta_plus = max(m_u.get(c, 0) - m_v.get(c, 0), 0)
            delta_minus = max(m_v.get(c, 0) - m_u.get(c, 0), 0)
            sum_c += (graph[c][v]['weight'] / w_uv) * delta_plus
            sum_c += (graph[c][u]['weight'] / w_uv) * delta_minus

        # Lower bound calculation
        lower_bound = 1 - sum_l - sum_r - sum_c
        lower_bounds[(u, v)] = lower_bound

    # Put result in shared queue for aggregation later
    result_queue.put((upper_bounds, lower_bounds))

def partition_edges(graph, num_workers):
    """
    Partition edges of the graph into batches for parallel processing.
    """
    edges = list(graph.edges(data=True))
    edge_batches = np.array_split(edges, num_workers)
    return edge_batches


def initialize_edge_weights(graph, default_weight=1.0):
    """
    Initialize edge weights for the graph. If an edge doesn't have a 'weight' attribute,
    assign it a default weight.
    """
    for u, v in graph.edges():
        if 'weight' not in graph[u][v]:
            graph[u][v]['weight'] = default_weight
    print(f"Initialized edge weights. Total edges: {graph.number_of_edges()}")


def ricci_flow_parallel(graph, iterations=20, epsilon=10, delta=1e-4, alpha=0.5, p=0, world_size=2):
    """
    Perform Ricci Flow with normalized weights and ORC bounds on a weighted graph using parallel processing.
    """
    initialize_edge_weights(graph)
    num_edges = graph.number_of_edges()
    orc_prev = {}

    for t in range(iterations):
        print(f"Iteration {t+1}/{iterations}")

        # Normalize edge weights
        total_weight = sum(graph[u][v]['weight'] for u, v in graph.edges())
        scaling_factor = num_edges / total_weight
        for u, v in graph.edges():
            graph[u][v]['weight'] *= scaling_factor

        # Compute probability measures m_u
        m_u_dict = compute_m_u(graph, alpha=alpha, p=p)

        # Partition edges for parallel processing
        edge_batches = partition_edges(graph, world_size)

        # Parallel computation setup
        result_queue = mp.Queue()
        processes = []

        for rank in range(world_size):
            edge_batch = edge_batches[rank]
            p = mp.Process(target=compute_orc_bounds, args=(rank, graph, edge_batch, result_queue, f'cuda:{rank}'))
            p.start()
            processes.append(p)

        # Collect results from all workers
        upper_bounds = {}
        lower_bounds = {}
        for _ in range(world_size):
            ubs, lbs = result_queue.get()
            upper_bounds.update(ubs)
            lower_bounds.update(lbs)

        for p in processes:
            p.join()

        # Compute ORC (average of bounds)
        orc = {}
        for edge in upper_bounds:
            orc[edge] = (upper_bounds[edge] + lower_bounds[edge]) / 2
            # if t==1:
            #     print(orc[edge], end = " ")

        # Update edge weights based on ORC
        for (u, v) in graph.edges():
            w_uv = graph[u][v]['weight']
            graph[u][v]['weight'] -= epsilon * orc[(u, v)] * w_uv

        # Check convergence using RMSE between previous and current ORC values
        if t > 0:  # Only compute RMSE after the first iteration
            orc_vals = torch.tensor([orc[edge] for edge in orc])
            orc_prev_vals = torch.tensor([orc_prev[edge] for edge in orc_prev])
            rmse = torch.sqrt(torch.mean((orc_vals - orc_prev_vals) ** 2)).item()

            print(f"RMSE between curvature matrices: {rmse:.6f}")
            if rmse < delta:
                print(f"Converged at iteration {t+1} with RMSE {rmse:.6f} < delta {delta}")
                break

            print(f"Range of ORC: {torch.min(orc_vals).item()} - {torch.max(orc_vals).item()}")

        # Store current ORC as previous ORC for the next iteration
        orc_prev = orc.copy()

    return graph


def save_adjacency_matrix(graph, output_file):
    """
    Save the final adjacency matrix (after Ricci flow) as a sparse CSR matrix.
    """
    row, col, data_values = [], [], []
    for u, v, data in graph.edges(data=True):
        row.append(u)
        col.append(v)
        data_values.append(data['weight'])
    num_nodes = graph.number_of_nodes()
    adj_matrix = csr_matrix((data_values, (row, col)), shape=(num_nodes, num_nodes))
    save_npz(output_file, adj_matrix)
    print(f"Saved adjacency matrix to {output_file}")


if __name__ == "__main__":
    import argparse
    from utils import Dataset  # Assuming you have a Dataset class for loading data

    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Accelerated Ricci Flow on GPU")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha parameter")
    parser.add_argument('--iterations', type=int, default=20, help="Maximum number of iterations")
    parser.add_argument('--epsilon', type=float, default=1, help="Learning rate")
    parser.add_argument('--delta', type=float, default=1e-4, help="Convergence threshold for RMSE")
    parser.add_argument('--output_dir', type=str, default='ricci_flow_data', help="Output directory")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for parallel computation")
    args = parser.parse_args()

    # Load dataset
    dataset = Dataset(name=args.dataset)
    graph = dataset.graph.to_networkx()

    # Convert the MultiDiGraph to a simple graph
    graph = convert_multidigraph_to_graph(graph)

    # Run Ricci Flow with GPU parallelization
    graph = ricci_flow_parallel(
        graph,
        iterations=args.iterations,
        epsilon=args.epsilon,
        delta=args.delta,
        alpha=args.alpha,
        world_size=args.world_size
    )

    # Save weighted adjacency matrix
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_ricci_flow_adj_matrix.npz")
    # save_adjacency_matrix(graph, output_file)
    
