"""Read-only GHRN checkpoint forward-repeat probe.

This intentionally has no optimizer, backward pass, checkpoint save, or metric
selection.  It determines whether the saved two-stage checkpoints reproduce
the same reduced graph and test scores across repeated GPU forwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from methods.ghrn.src.model import GHRNModel
from methods.ghrn.src.protocol import final_test_values
from methods.ghrn.src.runner import (
    edge_reduction_probability,
    load_frozen_hsmad_data,
    official_style_random_walk_update,
    prepare_training_graph,
)


def _hash_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _edge_hash(graph) -> str:
    src, dst = graph.edges(order='eid')
    return _hash_tensor(torch.stack((src, dst)))


def run_probe(run_dir: str | Path, repeats: int = 2) -> dict[str, object]:
    """Repeat frozen forward reconstruction; never trains or writes checkpoints."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    config = json.loads((run_dir / 'config_snapshot.json').read_text())
    raw, features, labels, masks, _ = load_frozen_hsmad_data(config['dataset'])
    device = torch.device('cuda')
    features, labels = features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    observations: list[dict[str, object]] = []
    for _ in range(repeats):
        graph = prepare_training_graph(raw).to(device)
        stage1 = GHRNModel(features.shape[1], int(config['hidden_dim']), int(config['order'])).to(device)
        stage2 = GHRNModel(features.shape[1], int(config['hidden_dim']), int(config['order'])).to(device)
        stage1.load_state_dict(torch.load(run_dir / 'checkpoint_bootstrap_auprc_best.pt', map_location=device)['model_state_dict'])
        stage2.load_state_dict(torch.load(run_dir / 'checkpoint_auprc_best.pt', map_location=device)['model_state_dict'])
        stage1.eval(); stage2.eval()
        with torch.no_grad():
            bootstrap = edge_reduction_probability(stage1(graph, features))
            reduced = official_style_random_walk_update(graph, bootstrap, float(config['delete_ratio']))
            logits = stage2(reduced, features)
            probability = edge_reduction_probability(logits)[:, 1]
        torch.cuda.synchronize(device)
        values = final_test_values(labels, probability, masks['test_mask'], float(metrics['threshold']))
        observations.append({
            'stage1_state_sha256': _state_hash(stage1),
            'stage2_state_sha256': _state_hash(stage2),
            'bootstrap_probability_sha256': _hash_tensor(bootstrap),
            'reduced_edge_sha256': _edge_hash(reduced),
            'reduced_edge_count': int(reduced.num_edges()),
            'final_logits_sha256': _hash_tensor(logits),
            'final_probability_sha256': _hash_tensor(probability),
            **values,
        })
    return {'run_dir': str(run_dir), 'source_threshold': metrics['threshold'], 'observations': observations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    result = run_probe(args.run_dir, args.repeats)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
