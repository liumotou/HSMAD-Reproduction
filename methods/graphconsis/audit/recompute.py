"""Independent checkpoint-only recomputation for GraphConsis candidate artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from methods.graphconsis.src.model import GraphConsisSingleRelationCandidate
from methods.graphconsis.src.protocol import select_validation_threshold, test_metrics
from methods.graphconsis.src.run import ROOT, build_padded_adjacency, evaluate_nodes, prepare_training_graph, write_json


FIELDS = ("f1_macro", "auroc", "threshold", "best_epoch", "predicted_anomaly_count")


def compare_metrics(original, recomputed):
    differences = {key: {"original": original[key], "recomputed": recomputed[key]}
                   for key in FIELDS if original[key] != recomputed[key]}
    return {"status": "recompute_match" if not differences else "recompute_mismatch", "differences": differences}


def recompute(output: Path):
    config=json.loads((output/"config_snapshot.json").read_text()); original=json.loads((output/"metrics.json").read_text())
    import dgl
    graph=prepare_training_graph(dgl.load_graphs(str(ROOT/config["dataset_file"]))[0][0])
    source,destination=graph.edges(order="eid")
    adjacency=build_padded_adjacency(source.cpu(),destination.cpu(),graph.num_nodes(),int(config["max_degree"]),int(config["seed"]))
    device=torch.device("cuda"); features=graph.ndata["feature"].float().to(device); labels=graph.ndata["label"].long().to(device)
    adjacency=adjacency.to(device); val_mask=graph.ndata["val_mask"].bool(); test_mask=graph.ndata["test_mask"].bool()
    val_nodes=val_mask.nonzero(as_tuple=False).flatten().to(device); test_nodes=test_mask.nonzero(as_tuple=False).flatten().to(device)
    model=GraphConsisSingleRelationCandidate(features.shape[1],int(config["hidden_dim"]),2,float(config["dropout"])).to(device)
    checkpoint_path=output/"checkpoint_validation_auprc_best.pt"; checkpoint=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_logits=evaluate_nodes(model,features,adjacency,val_nodes,int(config["batch_size"]),int(config["seed"])+100000)
    val_prob=torch.softmax(val_logits,1)[:,1]; global_val=torch.zeros(graph.num_nodes(),device=device); global_val[val_nodes]=val_prob
    threshold,_=select_validation_threshold(labels,global_val,val_mask.to(device))
    test_logits=evaluate_nodes(model,features,adjacency,test_nodes,int(config["batch_size"]),int(config["seed"])+200000)
    test_prob=torch.softmax(test_logits,1)[:,1]; global_test=torch.zeros(graph.num_nodes(),device=device); global_test[test_nodes]=test_prob
    values=test_metrics(labels,global_test,test_mask.to(device),threshold)
    values.update({"threshold":threshold,"best_epoch":int(checkpoint["epoch"]),
                   "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()})
    comparison=compare_metrics(original,values); result={**comparison,"original":{k:original[k] for k in FIELDS},"recomputed":values}
    audit_dir=output/"audit_recompute"; write_json(audit_dir/"recompute_result.json",result); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True); args=parser.parse_args()
    result=recompute(Path(args.output)); print(json.dumps(result,sort_keys=True)); raise SystemExit(0 if result["status"]=="recompute_match" else 2)


if __name__ == "__main__": main()
