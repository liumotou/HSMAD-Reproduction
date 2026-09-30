"""Dependency-free immutable architecture contract for GAT-v2-GADBench."""


def architecture_contract():
    return {
        "hidden_total_dim": 64,
        "num_heads": 4,
        "per_head_dim": 16,
        "num_gat_blocks": 2,
        "activation": "GELU",
        "output_dim": 2,
    }


def formal_execution_contract():
    return {
        "max_epoch": 200,
        "patience": 50,
        "early_stop_metric": "validation_AUPRC",
        "checkpoint_metric": "validation_AUPRC_best",
    }


def amazon_contract():
    return {
        "input_dim": 25,
        "uncovered_prefix_nodes": 3305,
        "training_graph_edges": 8808728,
        "train_mask_count": 3455,
        "val_mask_count": 1710,
        "test_mask_count": 3474,
    }


def yelp_contract():
    return {
        "input_dim": 32,
        "training_graph_edges": 7739912,
        "train_mask_count": 18381,
        "val_mask_count": 9099,
        "test_mask_count": 18474,
    }


def tolokers_contract():
    return {
        "input_dim": 10,
        "training_graph_edges": 1049758,
        "train_mask_count": 4703,
        "val_mask_count": 2328,
        "test_mask_count": 4727,
    }


def tfinance_contract():
    return {
        "input_dim": 10,
        "training_graph_edges": 42484443,
        "train_mask_count": 15742,
        "val_mask_count": 7792,
        "test_mask_count": 15823,
    }
