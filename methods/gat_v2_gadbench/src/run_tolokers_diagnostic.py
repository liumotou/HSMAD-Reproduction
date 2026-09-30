"""Isolated Tolokers seed-0 diagnostic for frozen GAT-v2-GADBench."""
import argparse,csv,hashlib,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import dgl,torch
import torch.nn.functional as F
from contracts import architecture_contract,tolokers_contract
from model import GADBenchGATV2
from run_diagnostic_full import checkpoint_test
from run_smoke import ROOT,best_threshold,file_sha256,framework,graph_sha256,protected_manifest,set_seed,split_metrics,tensor_sha256,write_json
from selection import update_selection

def manifest():
 m=protected_manifest()
 for rel in ("results/experiments/gat_v2_gadbench/weibo","results/experiments/gat_v2_gadbench/amazon","results/experiments/gat_v2_gadbench/yelp"):
  t=ROOT/rel
  if t.exists():
   for p in sorted(x for x in t.rglob("*") if x.is_file()):m["files"][str(p.relative_to(ROOT))]=file_sha256(p)
 m["manifest_sha256"]=hashlib.sha256(json.dumps(m["files"],sort_keys=True).encode()).hexdigest();return m
def probs(model,g,x):
 model.eval()
 with torch.no_grad():return torch.softmax(model(g,x),dim=1)[:,1]
def main():
 a=argparse.ArgumentParser();a.add_argument("--config",required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());out=ROOT/c["result_dir"]
 if out.exists():raise FileExistsError(f"Refusing to overwrite {out}")
 before=manifest();start=time.monotonic();out.mkdir(parents=True);write_json(out/"config_snapshot.json",c);shutil.copy2(cp,out/"config_source.json");write_json(out/"environment"/"framework.json",framework());write_json(out/"protected_manifest_before.json",before)
 set_seed(0);raw=dgl.load_graphs(str(ROOT/c["dataset_file"]))[0][0];g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));x,y=raw.ndata["feature"],raw.ndata["label"].long();rm={n:raw.ndata[n] for n in ("train_mask","val_mask","test_mask")};m={n:v.bool() for n,v in rm.items()};ct=tolokers_contract()
 hs={"dataset_file_sha256":file_sha256(ROOT/c["dataset_file"]),"feature_sha256":tensor_sha256(x),"label_sha256":tensor_sha256(y),**{f"{n}_sha256":tensor_sha256(v) for n,v in rm.items()},"raw_graph_edges_sha256":graph_sha256(raw),"training_graph_edges_sha256":graph_sha256(g),"model_py_sha256":file_sha256(ROOT/"methods/gat_v2_gadbench/src/model.py"),"run_tolokers_diagnostic_py_sha256":file_sha256(Path(__file__)),"selection_py_sha256":file_sha256(ROOT/"methods/gat_v2_gadbench/src/selection.py")}
 if any(hs[k]!=v for k,v in c["expected_hashes"].items()) or x.shape[1]!=ct["input_dim"] or g.num_edges()!=ct["training_graph_edges"] or any(int(m[n].sum())!=ct[f"{n}_count"] for n in m):raise RuntimeError("Frozen Tolokers inputs differ from HSMAD mask contract")
 tl=y[m["train_mask"]];weight=[1.0,int((tl==0).sum())/int(tl.sum())]
 if weight[1]!=c["expected_anomaly_weight"]:raise RuntimeError("Tolokers class weight mismatch")
 write_json(out/"preflight.json",{"passed":True,"raw_graph":{"nodes":raw.num_nodes(),"edges":raw.num_edges()},"training_graph":{"nodes":g.num_nodes(),"edges":g.num_edges()},"feature_shape":list(x.shape),"mask_counts":{n:int(v.sum()) for n,v in m.items()},"ignored_original_mask_fields":["train_masks","val_masks","test_masks"],"mask_source":c["mask_source"],"class_weight":weight,"hashes":hs,"model_contract":architecture_contract(),"edge_access":c["edge_access"]})
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");g,x,y=g.to(dev),x.to(dev),y.to(dev);m={n:v.to(dev) for n,v in m.items()};model=GADBenchGATV2(x.shape[1],64,4,0.,2).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);w=torch.tensor(weight,dtype=torch.float32,device=dev)
 if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(dev)
 s={"best_auprc":-1.,"auprc_best_epoch":None,"best_f1":-1.,"f1_best_epoch":None,"patience_counter":0};h=[];ac=out/"checkpoint_validation_auprc_best.pt";fc=out/"checkpoint_validation_f1_best_diagnostic.pt"
 for e in range(1,201):
  model.train();loss=F.cross_entropy(model(g,x)[m["train_mask"]],y[m["train_mask"]],weight=w);opt.zero_grad(set_to_none=True);loss.backward();opt.step();p=probs(model,g,x).cpu().numpy();idx=m["val_mask"].cpu().numpy();vl=y[m["val_mask"]].cpu().numpy();th,vf=best_threshold(vl,p[idx]);v=split_metrics(vl,p[idx],th);s,ai,fi=update_selection(s,e,v["auprc"],vf);payload={"epoch":e,"model_state_dict":model.state_dict(),"config":c,"validation":v}
  if ai:torch.save(payload,ac)
  if fi:torch.save(payload,fc)
  r={"epoch":e,"train_loss":float(loss.item()),"validation_f1_macro":vf,"validation_auroc":v["auroc"],"validation_auprc":v["auprc"],"validation_threshold":th,"auprc_best_epoch":s["auprc_best_epoch"],"f1_best_epoch":s["f1_best_epoch"],"patience_counter":s["patience_counter"],"learning_rate":.01,"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2 if torch.cuda.is_available() else 0.};h.append(r);print(json.dumps(r,sort_keys=True),flush=True)
  if s["patience_counter"]>50:break
 at,ft=checkpoint_test(ac,model,g,x,y,m),checkpoint_test(fc,model,g,x,y,m);after=manifest();M={"method":"GAT-v2-GADBench","protocol_version":c["protocol_version"],"dataset":"tolokers","seed":0,"run_type":"diagnostic_full","status":"diagnostic","actual_epochs":len(h),"stop_reason":"max_epoch_reached" if len(h)==200 else "validation_AUPRC_patience_exceeded","auprc_best_epoch":s["auprc_best_epoch"],"f1_best_epoch":s["f1_best_epoch"],"same_epoch":s["auprc_best_epoch"]==s["f1_best_epoch"],"auprc_best_checkpoint_test":at,"f1_best_checkpoint_test":ft,"paper":{"f1_macro":c["paper_f1_macro"],"auroc":c["paper_auroc"]},"paper_delta_auprc_best":{"f1_macro":at["f1_macro"]-c["paper_f1_macro"],"auroc":at["auroc"]-c["paper_auroc"]},"paper_delta_f1_best":{"f1_macro":ft["f1_macro"]-c["paper_f1_macro"],"auroc":ft["auroc"]-c["paper_auroc"]},"class_weight":weight,"edge_access":c["edge_access"],"hashes":hs,"wall_time_sec":time.monotonic()-start,"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2 if torch.cuda.is_available() else 0.,"protected_manifest_unchanged":before==after,"completed_at_utc":datetime.now(timezone.utc).isoformat()};write_json(out/"validation_history.json",h);write_json(out/"metrics.json",M);write_json(out/"protected_manifest_after.json",after)
 with (out/"runs.csv").open("w",newline="",encoding="utf-8") as f:
  fields=["method","protocol_version","dataset","seed","run_type","status","actual_epochs","stop_reason","auprc_best_epoch","f1_best_epoch","wall_time_sec","peak_gpu_mb","edge_access"];q=csv.DictWriter(f,fieldnames=fields);q.writeheader();q.writerow({k:M[k] for k in fields})
 write_json(out/"artifact_sha256s.json",{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()});print("TOLOKERS_DIAGNOSTIC_COMPLETE "+json.dumps(M,sort_keys=True),flush=True)
if __name__=="__main__":main()
