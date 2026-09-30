import os
import torch
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from dgl.data.utils import load_graphs
from utils import Dataset  # Importing Dataset class from utils.py for data loading
import time  # To measure time taken for curvature computation
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz


def convert_multidigraph_to_graph(multi_g):
    """
    Convert a MultiDiGraph to an undirected Graph by collapsing multiple edges into single edges.
    
    Parameters:
        multi_g (nx.MultiDiGraph): A MultiDiGraph to convert
    
    Returns:
        nx.Graph: An undirected Graph with the same nodes and one edge for each pair of nodes
    """
    graph = nx.Graph()

    print("Edges in MultiDiGraph", len(multi_g.edges()))

    # Add nodes from the MultiDiGraph
    graph.add_nodes_from(multi_g.nodes(data=True))

    # Add edges from the MultiDiGraph by collapsing multiple edges into single edges
    for u, v, data in multi_g.edges(data=True):
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, **data)
    print("Edges after converting to Graph", len(graph.edges()))
    return graph


def accelerated_ricci_approximation(graph, num_nodes):
    """
    Approximate ORC using the tighter lower bound in linear time and store in a sparse CSR matrix.
    
    Parameters:
    - graph: NetworkX graph
    - num_nodes: Number of nodes in the graph
    
    Returns:
    - curvature_matrix: CSR sparse matrix storing edge curvatures
    """
    row, col, data = [], [], []

    for u, v in graph.edges():
        d_u = graph.degree[u]
        d_v = graph.degree[v]
        num_triangles = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))

        # Compute the tighter lower bound
        def bound(d_u, d_v, num_triangles, d_min, d_max):
            term1 = max(0, 1 - (1 / d_u) - (1 / d_v) - (num_triangles / d_min))
            term2 = max(0, 1 - (1 / d_u) - (1 / d_v) - (num_triangles / d_max))
            return -term1 - term2 + (num_triangles / d_max)

        d_min = min(d_u, d_v)
        d_max = max(d_u, d_v)
        lower_bound = bound(d_u, d_v, num_triangles, d_min, d_max)
        upper_bound = num_triangles / d_max
        estimated_curvature = (lower_bound + upper_bound) / 2

        # Store curvature symmetrically for both (u, v) and (v, u)
        row.append(u)
        col.append(v)
        data.append(estimated_curvature)
        # row.append(v)
        # col.append(u)
        # data.append(estimated_curvature)
    
    # Convert to CSR matrix
    curvature_matrix = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    return curvature_matrix


def compute_and_store_curvature(data, dataset_name, save_dir='curvature_data', use_accelerated=False, compare_both=False, comparison_plot=False, save_matrix=False):
    """
    Compute and save Ollivier-Ricci curvature for nodes and edges in the dataset.
    
    Parameters:
    - data: Dataset object (GADBench format)
    - dataset_name: Name of the dataset (string)
    - save_dir: Directory where curvature data will be saved.
    - use_accelerated: Whether to use the accelerated (approximation) method for computing ORC.
    - compare_both: Whether to compute both exact and accelerated ORC and compare.
    
    Returns:
    - time_elapsed: Time taken to compute the curvature (in seconds).
    """
    # Create directory to save curvature data if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert the DGL graph to a NetworkX graph
    multi_di = data.graph.to_networkx()
    g = convert_multidigraph_to_graph(multi_di)
    num_nodes = g.number_of_nodes()
    print(f"Graph type for dataset {dataset_name}: {type(g)} with {num_nodes} nodes")

    # Initialize curvature matrices
    exact_curvature_matrix = None
    accelerated_curvature_matrix = None

    # Compute ORC
    print(f"Computing ORC for dataset: {dataset_name}")
    start_time = time.time()

    if compare_both or not use_accelerated:
        print("Using exact ORC computation (GraphRicciCurvature library).")
        orc = OllivierRicci(g, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()

        # Create matrix for exact curvature
        row, col, data = [], [], []
        for u, v, edge_data in orc.G.edges(data=True):
            row.append(u)
            col.append(v)
            data.append(edge_data['ricciCurvature'])
            # row.append(v)
            # col.append(u)
            # data.append(edge_data['ricciCurvature'])

        # Store as CSR matrix
        exact_curvature_matrix = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    if use_accelerated or compare_both:
        print("Using accelerated ORC computation (approximation).")
        accelerated_curvature_matrix = accelerated_ricci_approximation(g, num_nodes)
        print(accelerated_curvature_matrix.min(),accelerated_curvature_matrix.max() )
    
    time_elapsed = time.time() - start_time
    print(f"Time taken to compute ORC for {dataset_name}: {time_elapsed:.2f} seconds")

    if save_matrix:
        edge_curvature_file = os.path.join(save_dir, f"{dataset_name}_curvature_matrix.npz")
        save_npz(edge_curvature_file, accelerated_curvature_matrix)
        print(f"Curvature data saved for dataset: {dataset_name}")

    if compare_both and exact_curvature_matrix is not None and accelerated_curvature_matrix is not None:
        mse = mean_squared_error(exact_curvature_matrix.toarray(), accelerated_curvature_matrix.toarray())
        print(f"Mean Squared Error (MSE) between exact and accelerated ORC: {mse:.4f}")

    if comparison_plot:
        plot_orc_comparison(exact_curvature_matrix, accelerated_curvature_matrix, dataset_name, save_dir)
    
    return time_elapsed


def plot_orc_comparison(exact_curvature_matrix, accelerated_curvature_matrix, dataset_name, save_dir):
    """
    Plot a scatter plot comparing exact and accelerated ORC values for all edges.
    
    Parameters:
    - exact_curvature_matrix: CSR sparse matrix containing exact ORC values.
    - accelerated_curvature_matrix: CSR sparse matrix containing accelerated ORC values.
    """
    # Flatten both matrices to get curvature values for all edges
    exact_values = exact_curvature_matrix.toarray().flatten()
    accelerated_values = accelerated_curvature_matrix.toarray().flatten()

    # Filter out zero values (no edges)
    valid_indices = exact_values != 0
    exact_values = exact_values[valid_indices]
    accelerated_values = accelerated_values[valid_indices]

    # Plotting the scatter plot
    plt.figure(figsize=(8, 8))
    
    # Plot the 45-degree line (y = x) to represent the exact match
    plt.plot([-1, 1], [-1, 1], color='black', linestyle='--', label="Exact ORC (y = x)")
    
    # Scatter plot of approximate ORC vs exact ORC
    plt.scatter(exact_values, accelerated_values, alpha=0.2, color='blue', label="Approximate ORC", edgecolor='k')
    
    # Add labels and title
    plt.xlabel('Exact ORC (GraphRicciCurvature)')
    plt.ylabel('Approximate ORC (Accelerated)')
    plt.title('Comparison of Exact vs. Approximate ORC')
    
    # Set plot limits
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    
    # Add legend
    plt.legend()

    # Show grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_orc_comparison_plot.png"))


if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Compute and store Ollivier-Ricci Curvature for datasets")
    parser.add_argument('--datasets', nargs='+', default=['texas', 'cornell', 'wisconsin'], help="List of datasets to compute ORC for")
    parser.add_argument('--use_accelerated', action='store_true', help="Use the accelerated approximation method for ORC computation")
    parser.add_argument('--compare_both', action='store_true', help="Compute both exact and accelerated ORC and compare")
    parser.add_argument('--save_matrix', action='store_true', help="Save curvature matrix in case of accelerated computing")
    args = parser.parse_args()

    total_time_taken = {}

    for dataset_name in args.datasets:
        print(f"Loading dataset: {dataset_name}")

        # Load dataset using GADBench's Dataset class
        data = Dataset(name=dataset_name)

        # Compute and store curvature for the dataset, using user-specified options
        time_taken = compute_and_store_curvature(data, dataset_name, use_accelerated=args.use_accelerated, compare_both=args.compare_both, comparison_plot=args.compare_both, save_matrix=args.save_matrix)

        # Record the time taken for each dataset
        total_time_taken[dataset_name] = time_taken

    # Print the total time taken for each dataset
    print("\nSummary of ORC computation time:")
    for dataset_name, time_elapsed in total_time_taken.items():
        print(f"{dataset_name}: {time_elapsed:.2f} seconds")