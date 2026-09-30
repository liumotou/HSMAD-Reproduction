"""Independent Yelp formal runner for frozen GAT-v2-GADBench settings."""
import argparse, csv, hashlib, json, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import dgl, numpy as np, torch
import torch.nn.functional as F
from contracts import architecture_contract, yelp_contract, tolokers_contract, tfinance_contract
from model import GADBenchGATV2
from run_diagnostic_full import checkpoint_test
from run_smoke import ROOT, best_threshold, file_sha256, framework, graph_sha256, protected_manifest, set_seed, split_metrics, tensor_sha256, write_json
from selection import update_selection

def protected():
    m=protected_manifest()
    for rel in ("results/experiments/gat_v2_gadbench/weibo","results/experiments/gat_v2_gadbench/amazon","results/experiments/gat_v2_gadbench/yelp","results/experiments/gat_v2_gadbench/tolokers","results/experiments/gat_v2_gadbench/tfinance/protocol_v2_gadbench_hidden64/smoke","results/experiments/gat_v2_gadbench/tfinance/protocol_v2_gadbench_hidden64/diagnostic_full"):
        t=ROOT/rel
        if t.exists():
            for p in sorted(x for x in t.rglob("*") if x.is_file()): m["files"][str(p.relative_to(ROOT))]=file_sha256(p)
    m["manifest_sha256"]=hashlib.sha256(json.dumps(m["files"],sort_keys=True).encode()).hexdigest(); return m

def prob(model,g,x):
    model.eval()
    with torch.no_grad(): return torch.softmax(model(g,x),dim=1)[:,1]

def append_summary(d, m, c):
    fields=["method","protocol_version","dataset","seed","run_type","status","actual_epochs","stop_reason","auprc_best_epoch","wall_time_sec","peak_gpu_mb","f1_macro","auroc","threshold","edge_access","error"]
    p=d/"runs.csv"; exists=p.exists()
    with p.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists:w.writeheader()
        w.writerow({k:m.get(k,"") for k in fields})
    rows=list(csv.DictReader(p.open()))
    ok=[r for r in rows if r["status"]=="OK"]
    if len(ok)==10 and {int(r["seed"]) for r in ok}==set(range(10)):
        f1=np.array([float(r["f1_macro"]) for r in ok]); auc=np.array([float(r["auroc"]) for r in ok])
        s={"method":"GAT-v2-GADBench","dataset":c["dataset"],"protocol_version":c["protocol_version"],"n":10,"f1_macro_mean":float(f1.mean()),"f1_macro_std_sample":float(f1.std(ddof=1)),"auroc_mean":float(auc.mean()),"auroc_std_sample":float(auc.std(ddof=1)),"paper_f1_macro":c["paper_f1_macro"],"paper_auroc":c["paper_auroc"],"paper_delta_f1_macro":float(f1.mean()-c["paper_f1_macro"]),"paper_delta_auroc":float(auc.mean()-c["paper_auroc"])}
        write_json(d/"summary.json",s)
        with (d/"summary.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=s.keys());w.writeheader();w.writerow(s)

def main():
    a=argparse.ArgumentParser();a.add_argument("--config",required=True);a.add_argument("--seed",type=int,required=True);z=a.parse_args()
    if z.seed not in range(10):raise ValueError("Formal seeds are frozen to 0 through 9")
    cp=Path(z.config).resolve();c=json.loads(cp.read_text()); diag=json.loads((ROOT/f"methods/gat_v2_gadbench/configs/{c['dataset']}_protocol_v2_gadbench_hidden64_diagnostic_full.json").read_text())
    keys=("dataset_file","max_epoch","patience","h_feats","num_heads","per_head_dim","drop_rate","expected_anomaly_weight","optimizer","learning_rate","weight_decay","edge_access","graph_preprocess","expected_training_graph_edges","expected_hashes")
    if any(c[k]!=diag[k] for k in keys) or c.get("early_stop_protocol") != "validation_AUPRC_GADBench" or c.get("checkpoint_protocol") != "validation_AUPRC_best_GADBench" or c.get("threshold_protocol") != "validation_F1_macro_grid_0.05_to_0.95":raise RuntimeError("Formal configuration differs from completed frozen protocol")
    c["seed"]=z.seed;c["result_dir"]=c["result_dir_template"].format(seed=z.seed);out=ROOT/c["result_dir"];formal=out.parent
    if out.exists():raise FileExistsError(f"Refusing to overwrite {out}")
    before=protected();started=time.monotonic();out.mkdir(parents=True);write_json(out/"config_snapshot.json",c);shutil.copy2(cp,out/"config_source.json");write_json(out/"environment"/"framework.json",framework());write_json(out/"protected_manifest_before.json",before)
    try:
        set_seed(z.seed);raw=dgl.load_graphs(str(ROOT/c["dataset_file"]))[0][0];g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));x,y=raw.ndata["feature"],raw.ndata["label"].long();rm={n:raw.ndata[n] for n in ("train_mask","val_mask","test_mask")};m={n:v.bool() for n,v in rm.items()};ct={"yelp":yelp_contract,"tolokers":tolokers_contract,"tfinance":tfinance_contract}[c["dataset"]]()
        hs={"dataset_file_sha256":file_sha256(ROOT/c["dataset_file"]),"feature_sha256":tensor_sha256(x),"label_sha256":tensor_sha256(y),**{f"{n}_sha256":tensor_sha256(v) for n,v in rm.items()},"raw_graph_edges_sha256":graph_sha256(raw),"training_graph_edges_sha256":graph_sha256(g),"model_py_sha256":file_sha256(ROOT/"methods/gat_v2_gadbench/src/model.py"),"run_yelp_formal_py_sha256":file_sha256(Path(__file__)),"selection_py_sha256":file_sha256(ROOT/"methods/gat_v2_gadbench/src/selection.py")}
        if any(hs[k]!=v for k,v in c["expected_hashes"].items()) or x.shape[1]!=ct["input_dim"] or g.num_edges()!=ct["training_graph_edges"] or any(int(m[n].sum())!=ct[f"{n}_count"] for n in m):raise RuntimeError("Frozen Yelp inputs differ from contract")
        tl=y[m["train_mask"]];weight=[1.0,int((tl==0).sum())/int(tl.sum())]
        if weight[1]!=c["expected_anomaly_weight"]:raise RuntimeError("Yelp class weight mismatch")
        write_json(out/"preflight.json",{"passed":True,"raw_graph":{"nodes":raw.num_nodes(),"edges":raw.num_edges()},"training_graph":{"nodes":g.num_nodes(),"edges":g.num_edges()},"feature_shape":list(x.shape),"mask_counts":{n:int(v.sum()) for n,v in m.items()},"class_weight":weight,"gpu_free_mb_before":torch.cuda.mem_get_info()[0]/1024**2 if torch.cuda.is_available() else None,"hashes":hs,"model_contract":architecture_contract(),"edge_access":c["edge_access"]})
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");g,x,y=g.to(dev),x.to(dev),y.to(dev);m={n:v.to(dev) for n,v in m.items()};model=GADBenchGATV2(x.shape[1],64,4,0.,2).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);w=torch.tensor(weight,dtype=torch.float32,device=dev)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(dev)
        state={"best_auprc":-1.,"auprc_best_epoch":None,"best_f1":-1.,"f1_best_epoch":None,"patience_counter":0};hist=[];ck=out/"checkpoint_validation_auprc_best.pt"
        for e in range(1,201):
            model.train();loss=F.cross_entropy(model(g,x)[m["train_mask"]],y[m["train_mask"]],weight=w);opt.zero_grad(set_to_none=True);loss.backward();opt.step();p=prob(model,g,x).cpu().numpy();idx=m["val_mask"].cpu().numpy();vl=y[m["val_mask"]].cpu().numpy();th,vf=best_threshold(vl,p[idx]);v=split_metrics(vl,p[idx],th);state,improved,_=update_selection(state,e,v["auprc"],vf)
            if improved:torch.save({"epoch":e,"model_state_dict":model.state_dict(),"config":c,"validation":v},ck)
            r={"epoch":e,"train_loss":float(loss.item()),"validation_f1_macro":vf,"validation_auroc":v["auroc"],"validation_auprc":v["auprc"],"validation_threshold":th,"auprc_best_epoch":state["auprc_best_epoch"],"patience_counter":state["patience_counter"],"learning_rate":.01,"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2 if torch.cuda.is_available() else 0.};hist.append(r);print(json.dumps(r,sort_keys=True),flush=True)
            if state["patience_counter"]>50:break
        t=checkpoint_test(ck,model,g,x,y,m);metrics={"method":"GAT-v2-GADBench","protocol_version":c["protocol_version"],"dataset":c["dataset"],"seed":z.seed,"run_type":"formal","status":"OK","actual_epochs":len(hist),"stop_reason":"max_epoch_reached" if len(hist)==200 else "validation_AUPRC_patience","auprc_best_epoch":state["auprc_best_epoch"],"f1_macro":t["f1_macro"],"auroc":t["auroc"],"auprc":t["auprc"],"threshold":t["threshold"],"wall_time_sec":time.monotonic()-started,"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2 if torch.cuda.is_available() else 0.,"edge_access":c["edge_access"],"hashes":hs,"error":""};write_json(out/"validation_history.json",hist)
    except Exception as exc:
        metrics={"method":"GAT-v2-GADBench","protocol_version":c["protocol_version"],"dataset":c["dataset"],"seed":z.seed,"run_type":"formal","status":"ERROR","actual_epochs":0,"stop_reason":"ERROR","auprc_best_epoch":"","wall_time_sec":time.monotonic()-started,"peak_gpu_mb":0.,"f1_macro":"","auroc":"","threshold":"","edge_access":c["edge_access"],"error":repr(exc)};raise
    finally:
        after=protected();metrics["protected_manifest_unchanged"]=before==after;write_json(out/"metrics.json",metrics);write_json(out/"protected_manifest_after.json",after);write_json(out/"artifact_sha256s.json",{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()});append_summary(formal,metrics,c)
    print("YELP_FORMAL_COMPLETE "+json.dumps(metrics,sort_keys=True),flush=True)
if __name__=="__main__":main()
