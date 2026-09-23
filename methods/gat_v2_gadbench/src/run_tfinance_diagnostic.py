"""Full seed-0 diagnostic, without adapting frozen T-Finance GAT-v2 settings."""
import argparse,csv,hashlib,json,shutil,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import dgl,torch
import torch.nn.functional as F
from model import GADBenchGATV2
from run_diagnostic_full import checkpoint_test
from run_smoke import ROOT,best_threshold,file_sha256,framework,graph_sha256,protected_manifest,set_seed,split_metrics,tensor_sha256,write_json
from selection import update_selection
def manifest():
 m=protected_manifest()
 for rel in ("results/experiments/gat_v2_gadbench/weibo","results/experiments/gat_v2_gadbench/amazon","results/experiments/gat_v2_gadbench/yelp","results/experiments/gat_v2_gadbench/tolokers","results/experiments/gat_v2_gadbench/tfinance/protocol_v2_gadbench_hidden64/smoke"):
  t=ROOT/rel
  if t.exists():
   for p in sorted(x for x in t.rglob("*") if x.is_file()):m["files"][str(p.relative_to(ROOT))]=file_sha256(p)
 m["manifest_sha256"]=hashlib.sha256(json.dumps(m["files"],sort_keys=True).encode()).hexdigest();return m
def prob(model,g,x):
 model.eval()
 with torch.no_grad():return torch.softmax(model(g,x),dim=1)[:,1]
def main():
 a=argparse.ArgumentParser();a.add_argument("--config",required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());out=ROOT/c["result_dir"]
 if out.exists():raise FileExistsError(f"Refusing to overwrite {out}")
 before=manifest();start=time.monotonic();out.mkdir(parents=True);write_json(out/"config_snapshot.json",c);shutil.copy2(cp,out/"config_source.json");write_json(out/"environment"/"framework.json",framework());write_json(out/"protected_manifest_before.json",before);h=[];stage="load"
 try:
  set_seed(0);raw=dgl.load_graphs(str(ROOT/c["dataset_file"]))[0][0];stage="graph_preprocess";g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));x,y=raw.ndata["feature"],raw.ndata["label"].long();rm={n:raw.ndata[n] for n in ("train_mask","val_mask","test_mask")};m={n:v.bool() for n,v in rm.items()};hs={"dataset_file_sha256":file_sha256(ROOT/c["dataset_file"]),"feature_sha256":tensor_sha256(x),"label_sha256":tensor_sha256(y),**{f"{n}_sha256":tensor_sha256(v) for n,v in rm.items()},"raw_graph_edges_sha256":graph_sha256(raw),"training_graph_edges_sha256":graph_sha256(g),"model_py_sha256":file_sha256(ROOT/"methods/gat_v2_gadbench/src/model.py"),"runner_py_sha256":file_sha256(Path(__file__))}
  if any(hs[k]!=v for k,v in c["expected_hashes"].items()) or g.num_edges()!=c["expected_training_graph_edges"]:raise RuntimeError("Frozen T-Finance inputs differ from contract")
  tl=y[m["train_mask"]];weight=[1.,int((tl==0).sum())/int(tl.sum())]
  if weight[1]!=c["expected_anomaly_weight"]:raise RuntimeError("T-Finance class weight mismatch")
  write_json(out/"preflight.json",{"passed":True,"raw_graph":{"nodes":raw.num_nodes(),"edges":raw.num_edges()},"training_graph":{"nodes":g.num_nodes(),"edges":g.num_edges()},"feature_shape":list(x.shape),"mask_counts":{n:int(v.sum()) for n,v in m.items()},"class_weight":weight,"hashes":hs,"edge_access":c["edge_access"]})
  stage="device_transfer";dev=torch.device("cuda");g,x,y=g.to(dev),x.to(dev),y.to(dev);m={n:v.to(dev) for n,v in m.items()};model=GADBenchGATV2(x.shape[1],64,4,0.,2).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);w=torch.tensor(weight,dtype=torch.float32,device=dev);torch.cuda.reset_peak_memory_stats(dev);s={"best_auprc":-1.,"auprc_best_epoch":None,"best_f1":-1.,"f1_best_epoch":None,"patience_counter":0};ac=out/"checkpoint_validation_auprc_best.pt";fc=out/"checkpoint_validation_f1_best_diagnostic.pt"
  for e in range(1,201):
   stage=f"epoch_{e}";model.train();loss=F.cross_entropy(model(g,x)[m["train_mask"]],y[m["train_mask"]],weight=w);opt.zero_grad(set_to_none=True);loss.backward();opt.step();p=prob(model,g,x).cpu().numpy();idx=m["val_mask"].cpu().numpy();vl=y[m["val_mask"]].cpu().numpy();th,vf=best_threshold(vl,p[idx]);v=split_metrics(vl,p[idx],th);s,ai,fi=update_selection(s,e,v["auprc"],vf);payload={"epoch":e,"model_state_dict":model.state_dict(),"config":c,"validation":v}
   if ai:torch.save(payload,ac)
   if fi:torch.save(payload,fc)
   r={"epoch":e,"train_loss":float(loss.item()),"validation_f1_macro":vf,"validation_auroc":v["auroc"],"validation_auprc":v["auprc"],"validation_threshold":th,"auprc_best_epoch":s["auprc_best_epoch"],"f1_best_epoch":s["f1_best_epoch"],"patience_counter":s["patience_counter"],"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2};h.append(r);print(json.dumps(r,sort_keys=True),flush=True)
   if s["patience_counter"]>50:break
  at,ft=checkpoint_test(ac,model,g,x,y,m),checkpoint_test(fc,model,g,x,y,m);M={"status":"diagnostic","run_type":"diagnostic_full","dataset":"tfinance","seed":0,"actual_epochs":len(h),"stop_reason":"max_epoch_reached" if len(h)==200 else "validation_AUPRC_patience_exceeded","auprc_best_epoch":s["auprc_best_epoch"],"f1_best_epoch":s["f1_best_epoch"],"same_epoch":s["auprc_best_epoch"]==s["f1_best_epoch"],"auprc_best_checkpoint_test":at,"f1_best_checkpoint_test":ft,"paper":{"f1_macro":c["paper_f1_macro"],"auroc":c["paper_auroc"]},"paper_delta_auprc_best":{"f1_macro":at["f1_macro"]-c["paper_f1_macro"],"auroc":at["auroc"]-c["paper_auroc"]},"paper_delta_f1_best":{"f1_macro":ft["f1_macro"]-c["paper_f1_macro"],"auroc":ft["auroc"]-c["paper_auroc"]},"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2,"wall_time_sec":time.monotonic()-start,"edge_access":c["edge_access"],"hashes":hs}
 except torch.cuda.OutOfMemoryError:
  M={"status":"OOM","run_type":"diagnostic_full","dataset":"tfinance","seed":0,"stage":stage,"error":traceback.format_exc(),"actual_epochs":len(h),"peak_gpu_mb":torch.cuda.max_memory_allocated()/1024**2,"allocated_gpu_mb":torch.cuda.memory_allocated()/1024**2,"reserved_gpu_mb":torch.cuda.memory_reserved()/1024**2,"wall_time_sec":time.monotonic()-start,"edge_access":c["edge_access"]}
 except Exception:
  M={"status":"ERROR","run_type":"diagnostic_full","dataset":"tfinance","seed":0,"stage":stage,"error":traceback.format_exc(),"actual_epochs":len(h),"wall_time_sec":time.monotonic()-start,"edge_access":c["edge_access"]}
 finally:
  write_json(out/"validation_history.json",h);after=manifest();M["protected_manifest_unchanged"]=before==after;write_json(out/"metrics.json",M);write_json(out/"protected_manifest_after.json",after);write_json(out/"artifact_sha256s.json",{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()});
  with (out/"runs.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["dataset","seed","run_type","status","stage","actual_epochs","wall_time_sec","peak_gpu_mb","edge_access","error"]);w.writeheader();w.writerow({k:M.get(k,"") for k in w.fieldnames})
 print("TFINANCE_DIAGNOSTIC_COMPLETE "+json.dumps(M,sort_keys=True),flush=True)
if __name__=="__main__":main()
