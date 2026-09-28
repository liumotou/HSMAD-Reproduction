import torch
import os
from pygod.utils import load_data
from dgl import save_graphs
from dgl.data.utils import save_info
import dgl
import numpy as np
import torch
import dgl
from sklearn.model_selection import StratifiedKFold
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, SquirrelDataset, ChameleonDataset, FraudAmazonDataset
from dgl.data import TexasDataset, CornellDataset, WisconsinDataset, ActorDataset

# List of datasets to process
DATASETS = ['cora', 'citeseer', 'actor', 'wisconsin', 'cornell', 'chameleon', 'squirrel', 'actor']

# Define the output directory
OUT_DIR = '../datasets'

# def save_dataset(graph, name, labels, masks):
#     """
#     Save the processed graph in DGL format along with masks and labels.
#     """
#     if not os.path.exists(OUT_DIR):
#         os.makedirs(OUT_DIR)

#     # Add masks to the graph for training, validation, and testing
#     graph.ndata['train_masks'] = masks['train_mask']
#     graph.ndata['val_masks'] = masks['val_mask']
#     graph.ndata['test_masks'] = masks['test_mask']
#     graph.ndata['label'] = labels

#     # Save the graph
#     dataset_path = os.path.join(OUT_DIR, f'{name}')
#     save_graphs(dataset_path, [graph])

#     print(f"Dataset {name} processed and saved.")

def save_dataset(graph, name, labels, masks, label_number = None):
    """
    Save the processed graph in DGL format along with masks and labels.
    """
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Add masks to the graph for training, validation, and testing
    graph.ndata['train_masks'] = masks['train_mask']
    graph.ndata['val_masks'] = masks['val_mask']
    graph.ndata['test_masks'] = masks['test_mask']
    graph.ndata['label'] = labels

    # Save the graph
    dataset_path = os.path.join(OUT_DIR, f'{name}_{label_number}')
    save_graphs(dataset_path, [graph])

    print(f"Dataset {name} processed and saved.")

def create_anomaly_classification_problem(graph, labels, dataset_name):
    """
    Convert the dataset into a one-vs-rest classification problem where the minority class is treated as anomalies.
    Print the percentage of anomalies (minority class).
    """
    # Find the minority class (the smallest class)
    # class_counts = torch.bincount(labels)
    # minority_class = torch.argmin(class_counts).item()
    
    unique_labels = np.unique(labels)
    # Convert to binary anomaly detection problem: 1 for anomaly, 0 for non-anomalous
    
    anomaly_labels_list = []
    for minority_class in unique_labels:
        anomaly_labels = (labels == minority_class).long()
        # Print the percentage of anomalies
        percentage_anomalies = (anomaly_labels.sum().item() / len(labels)) * 100
        print(f"{dataset_name} - Minority class {minority_class} treated as anomaly class.")
        print(f"Percentage of anomalies: {percentage_anomalies:.2f}%")
        anomaly_labels_list.append(anomaly_labels)
        
    return anomaly_labels_list


# def create_anomaly_classification_problem(graph, labels, dataset_name):
#     """
#     Convert the dataset into a one-vs-rest classification problem where the minority class is treated as anomalies.
#     Print the percentage of anomalies (minority class).
#     """
#     # Find the minority class (the smallest class)
#     class_counts = torch.bincount(labels)
#     minority_class = torch.argmin(class_counts).item()
    
#     # Convert to binary anomaly detection problem: 1 for anomaly, 0 for non-anomalous
#     anomaly_labels = (labels == minority_class).long()
    
#     # Print the percentage of anomalies
#     percentage_anomalies = (anomaly_labels.sum().item() / len(labels)) * 100
#     print(f"{dataset_name} - Minority class {minority_class} treated as anomaly class.")
#     print(f"Percentage of anomalies: {percentage_anomalies:.2f}%")
    
#     return anomaly_labels


def load_dgl_dataset(dataset_name):
    """
    Load the dataset from DGL's built-in datasets.
    Supported datasets: Cora, Citeseer, Pubmed
    """
    if dataset_name == 'cora':
        dataset = CoraGraphDataset()
    elif dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif dataset_name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif dataset_name == 'squirrel':
        dataset = SquirrelDataset()
    elif dataset_name == 'chameleon':
        dataset = ChameleonDataset()
    elif dataset_name == 'amazon_fraud':
        dataset = FraudAmazonDataset()
    elif dataset_name == 'texas':
        dataset = TexasDataset()
    elif dataset_name == 'cornell':
        dataset = CornellDataset()
    elif dataset_name == 'wisconsin':
        dataset = WisconsinDataset()
    elif dataset_name == 'actor':
        dataset = ActorDataset()
    else:
        raise ValueError(f"Unsupported DGL dataset: {dataset_name}")
    
    graph = dataset[0]
    labels = graph.ndata['label']
    
    return graph, labels

def process_graph(py_g_graph, num_folds=20, dataset_name = "disney"):
    """
    Convert the PyGOD format graph to DGL format and extract necessary information.
    Ensure that splits are stratified based on labels.

    Parameters:
    - py_g_graph: PyGOD graph object
    - num_folds: Number of splits (default is 20, to match existing datasets)

    Returns:
    - g_dgl: DGL graph
    - labels: Node labels
    - split_masks: Dictionary containing 'train_mask', 'val_mask', 'test_mask'
    """
    if isinstance(py_g_graph, dgl.DGLGraph) and all(k in py_g_graph.ndata for k in ['train_mask', 'val_mask', 'test_mask', 'label']):
        # Already processed, return as is
        g_dgl = py_g_graph
        labels = g_dgl.ndata['label']
        train_mask = g_dgl.ndata['train_mask']
        val_mask = g_dgl.ndata['val_mask']
        test_mask = g_dgl.ndata['test_mask']
        try:
            g_dgl.ndata['feature'] = g_dgl.ndata['feat']
        except:
            pass
    else:
        # Convert PyG graph to DGL graph
        g_dgl = dgl.graph((py_g_graph.edge_index[0], py_g_graph.edge_index[1]))

        # Add node features
        if 'x' in py_g_graph:
            g_dgl.ndata['feature'] = py_g_graph.x

        # Add edge weights if available
        if 'edge_weight' in py_g_graph:
            g_dgl.edata['weight'] = py_g_graph.edge_weight

        # Add labels
        if 'y' in py_g_graph:
            labels = py_g_graph.y
            if dataset_name == "inj_cora":
                labels = py_g_graph.y.bool()
        else:
            num_nodes = g_dgl.num_nodes()
            labels = torch.zeros(num_nodes, dtype=torch.long)  # Placeholder labels if not provided

        num_nodes = g_dgl.num_nodes()

        # Initialize empty masks for train, val, and test)
        train_mask = torch.zeros((num_nodes, num_folds), dtype=torch.bool)
        val_mask = torch.zeros((num_nodes, num_folds), dtype=torch.bool)
        test_mask = torch.zeros((num_nodes, num_folds), dtype=torch.bool)

        # Stratified split for each fold
        skf = StratifiedKFold(n_splits=num_folds)
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(torch.zeros(num_nodes), labels)):
            # Split train_val_idx into train and val
            train_idx, val_idx = train_val_idx[:int(0.8 * len(train_val_idx))], train_val_idx[int(0.8 * len(train_val_idx)):]

            # Set the masks for the current fold
            train_mask[train_idx, fold] = True
            val_mask[val_idx, fold] = True
            test_mask[test_idx, fold] = True

    # Return processed graph and masks
    return g_dgl, labels, {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}


def load_and_process_dataset(dataset_name):
    """
    Load dataset using PyGOD or DGL, process it, and save it in DGL-compatible format.
    """
    print(f"Processing {dataset_name} dataset...")

    if dataset_name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'amazon_fraud', 'cornell', 'texas', 'wisconsin', 'actor']:
        # Load from DGL built-in datasets
        dgl_graph, labels = load_dgl_dataset(dataset_name)
        
        # Convert to one-vs-rest anomaly classification problem
        labels_list = create_anomaly_classification_problem(dgl_graph, labels, dataset_name)
        # labels = create_anomaly_classification_problem(dgl_graph, labels, dataset_name)
        
        # Process and save
        for idx, labels in enumerate(labels_list):
            num_folds = 20
            masks = process_graph(dgl_graph, num_folds=num_folds, dataset_name=dataset_name)[2]
            save_dataset(dgl_graph, dataset_name, labels, masks, idx)
    else:
        # Load from PyGOD for other datasets
        pyg_graph = load_data(dataset_name)

        # Process the PyGOD graph to DGL format
        dgl_graph, labels, masks = process_graph(pyg_graph, dataset_name=dataset_name)

        # Save the processed graph, labels, and masks in DGL format
        save_dataset(dgl_graph, dataset_name, labels, masks)

    print(f"Dataset {dataset_name} processed and saved.")

if __name__ == "__main__":
    for dataset in DATASETS:
        load_and_process_dataset(dataset)