from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.dsgad.src.protocol import prepare_training_graph
from methods.dsgad.src.protocol import setup_seed
from methods.dsgad.src.runner import load_data
from methods.spacegnn.official_snapshot.model import SpaceGNN
from methods.spacegnn_hsmad.src.protocol import test_values, validation_values
from methods.project_paths import project_root

ROOT = project_root()


def candidate_config(dataset: str, run_type: str = "formal") -> dict:
    if dataset not in {"weibo", "tolokers", "amazon", "tfinance"}:
        raise ValueError(dataset)
    official = {
        "weibo": (6, 0.0, 0.5, 1.0),
        "tolokers": (1, 0.05, 0.0, 0.5),
        "amazon": (3, 0.05, 0.0, 0.0),
        "tfinance": (3, 0.1, 1.0, 1.0),
    }
    layer_num, dropout, alpha, beta = official[dataset]
    rng = random.Random(1)
    stdneg, stdpos = rng.uniform(0.01, 0.02), rng.uniform(0.01, 0.02)
    return {"dataset": dataset, "run_type": run_type, "protocol_version": "spacegnn_hsmad_candidate",
            "positioning": "candidate_protocol_not_author_exact",
            "official_commit": "921c03ff879b239dab9b319b296fef3bc3bda2d2",
            "hidden_dim": 64, "learning_rate": 0.01, "optimizer": "Adam", "weight_decay": 0.0,
            "layer_num": layer_num, "dropout": dropout, "alpha": alpha, "beta": beta,
            "stdneg": stdneg, "stdpos": stdpos, "batch_size": 50,
            "max_epoch": 5 if run_type == "smoke" else 25,
            "checkpoint_metric": "validation_auroc_plus_f1_macro",
            "threshold_protocol": "validation_F1_macro_grid_0.05_to_0.95",
            "graph_preprocess": "to_bidirected->remove_self_loop->add_self_loop"}


def _propagate(features, graph):
    with graph.local_scope():
        graph.ndata["h"] = features
        graph.update_all(dgl.function.copy_u("h", "m"), dgl.function.mean("m", "h"))
        return graph.ndata["h"]


def prepare_spacegnn_graph(raw, layer_num: int):
    graph = prepare_training_graph(raw)
    graph.ndata["label"] = raw.ndata["label"].reshape(-1).long().clone()
    for name in ("train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = raw.ndata[name].bool().clone()
    features = raw.ndata["feature"].float().clone()
    graph.ndata["feature_0"] = features
    for index in range(1, layer_num):
        features = _propagate(features, graph)
        graph.ndata[f"feature_{index}"] = features
    return graph.long()


def build_model(input_dim, hidden_dim, layer_num, dropout, cneg, cpos):
    return SpaceGNN(input_dim, hidden_dim, 2, layer_num, dropout, cneg, cpos)


def checkpoint_score(validation):
    return float(validation["auroc"]) + float(validation["f1_macro"])


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _hash_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _probability(model, graph, sampler, node_ids, device, alpha, beta):
    loader = DataLoader(node_ids, batch_size=100000, shuffle=False)
    values, labels = [], []
    model.eval()
    with torch.no_grad():
        for index in loader:
            _, _, blocks = sampler.sample_blocks(graph, index)
            blocks = [block.to(device) for block in blocks]
            p1, p2, p3 = model(blocks)
            values.append(((1-beta)*((1-alpha)*p1.exp()[:,1] + alpha*p2.exp()[:,1]) + beta*p3.exp()[:,1]).cpu())
            labels.append(blocks[-1].dstdata["label"].cpu())
    return torch.cat(values), torch.cat(labels)


def run_one(dataset, seed, run_type):
    cfg = candidate_config(dataset, run_type)
    output = ROOT / "results/experiments/spacegnn_hsmad" / dataset / "spacegnn_hsmad_candidate" / run_type / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    setup_seed(seed)
    raw, raw_path = load_data(dataset); graph = prepare_spacegnn_graph(raw, cfg["layer_num"])
    masks = {name: graph.ndata[name].bool() for name in ("train_mask","val_mask","test_mask")}
    meta = {"dataset_sha256":_hash_file(raw_path), "feature_sha256":_hash_tensor(raw.ndata["feature"]),
            "label_sha256":_hash_tensor(raw.ndata["label"].reshape(-1)),
            "mask_sha256":{k:_hash_tensor(v) for k,v in masks.items()}, "mask_counts":{k:int(v.sum()) for k,v in masks.items()},
            "raw_nodes":raw.num_nodes(), "raw_edges":raw.num_edges(), "training_edges":graph.num_edges(),
            "runner_sha256":_hash_file(__file__), "model_sha256":_hash_file(Path(__file__).parents[2]/"spacegnn/official_snapshot/model.py")}
    _dump(output/"config_snapshot.json", {**cfg,"seed":seed}); _dump(output/"preflight.json", {"passed":True,"meta":meta})
    device=torch.device("cuda"); sampler=dgl.dataloading.MultiLayerFullNeighborSampler(1)
    train_ids=torch.nonzero(masks["train_mask"],as_tuple=False).reshape(-1); val_ids=torch.nonzero(masks["val_mask"],as_tuple=False).reshape(-1); test_ids=torch.nonzero(masks["test_mask"],as_tuple=False).reshape(-1)
    train_loader=DataLoader(train_ids,batch_size=cfg["batch_size"],shuffle=True,drop_last=True)
    cneg=torch.FloatTensor(cfg["layer_num"]).normal_(-.1,cfg["stdneg"]); cpos=torch.FloatTensor(cfg["layer_num"]).normal_(.1,cfg["stdpos"])
    model=build_model(raw.ndata["feature"].shape[1],cfg["hidden_dim"],cfg["layer_num"],cfg["dropout"],cneg,cpos).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    best,best_epoch,best_state,history=float("-inf"),0,None,[]; torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    for epoch in range(1,cfg["max_epoch"]+1):
        model.train(); loss_sum=0.; seen=0
        for index in train_loader:
            _,_,blocks=sampler.sample_blocks(graph,index); blocks=[b.to(device) for b in blocks]
            p1,p2,p3=model(blocks); mixed=(1-cfg["beta"])*((1-cfg["alpha"])*p1+cfg["alpha"]*p2)+cfg["beta"]*p3
            target=blocks[-1].dstdata["label"]; loss=F.nll_loss(mixed,target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach().cpu())*target.numel(); seen+=target.numel()
        probability,labels=_probability(model,graph,sampler,val_ids,device,cfg["alpha"],cfg["beta"])
        threshold,f1,auroc=validation_values(labels,probability,torch.ones_like(labels,dtype=torch.bool)); val={"threshold":threshold,"f1_macro":f1,"auroc":auroc}
        history.append({"epoch":epoch,"train_loss":loss_sum/seen,"validation_f1_macro":f1,"validation_auroc":auroc,"threshold":threshold})
        score=checkpoint_score(val)
        if score>=best: best,best_epoch,best_state=score,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    vp,vl=_probability(model,graph,sampler,val_ids,device,cfg["alpha"],cfg["beta"]); threshold,vf,va=validation_values(vl,vp,torch.ones_like(vl,dtype=torch.bool))
    tp,tl=_probability(model,graph,sampler,test_ids,device,cfg["alpha"],cfg["beta"]); metrics=test_values(tl,tp,torch.ones_like(tl,dtype=torch.bool),threshold)
    metrics.update({"method":"SpaceGNN-official-topology-h64","dataset":dataset,"seed":seed,"run_type":run_type,"status":"smoke" if run_type=="smoke" else "OK","positioning":cfg["positioning"],"best_epoch":best_epoch,"actual_epochs":epoch,"validation_auroc":va,"validation_f1_macro":vf,"threshold":threshold,"wall_time_sec":time.monotonic()-started,"peak_gpu_mb":torch.cuda.max_memory_allocated()/1024**2,"hashes":meta,"edge_access":"graph_edges_required"})
    torch.save({"model_state_dict":best_state,"config":cfg,"best_epoch":best_epoch,"threshold":threshold},output/"checkpoint_best.pt")
    _dump(output/"validation_history.json",history); _dump(output/"metrics.json",metrics); _dump(output/"artifact_sha256s.json",{p.name:_hash_file(p) for p in output.iterdir() if p.is_file()})
    return metrics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",required=True,choices=("weibo","tolokers","amazon","tfinance")); parser.add_argument("--seeds",default="0"); parser.add_argument("--run-type",default="smoke",choices=("smoke","diagnostic","formal")); args=parser.parse_args(); rows=[]
    for item in args.seeds.split(","):
        seed=int(item)
        try: rows.append(run_one(args.dataset,seed,args.run_type))
        except Exception as error: rows.append({"dataset":args.dataset,"seed":seed,"run_type":args.run_type,"status":"ERROR","error":repr(error)})
        base=ROOT/"results/experiments/spacegnn_hsmad"/args.dataset/"spacegnn_hsmad_candidate"/args.run_type; fields=sorted({k for row in rows for k in row})
        with (base/"runs.csv").open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,sort_keys=True))


if __name__ == "__main__": main()
