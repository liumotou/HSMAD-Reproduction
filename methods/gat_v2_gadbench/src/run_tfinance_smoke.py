"""Five-epoch, no-adaptation T-Finance GAT-v2 memory smoke."""
import argparse,csv,hashlib,json,shutil,time,traceback
from pathlib import Path
import dgl,torch
import torch.nn.functional as F
from model import GADBenchGATV2
from run_smoke import ROOT,best_threshold,file_sha256,framework,graph_sha256,protected_manifest,set_seed,split_metrics,tensor_sha256,write_json
def main():
 a=argparse.ArgumentParser();a.add_argument("--config",required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());out=ROOT/c["result_dir"]
 if out.exists():raise FileExistsError(f"Refusing to overwrite {out}")
 before=protected_manifest();start=time.monotonic();out.mkdir(parents=True);write_json(out/"config_snapshot.json",c);shutil.copy2(cp,out/"config_source.json");write_json(out/"environment"/"framework.json",framework());write_json(out/"protected_manifest_before.json",before);history=[];stage="load"
 try:
  set_seed(0);raw=dgl.load_graphs(str(ROOT/c["dataset_file"]))[0][0];stage="graph_preprocess";g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));x,y=raw.ndata["feature"],raw.ndata["label"].long();rm={n:raw.ndata[n] for n in ("train_mask","val_mask","test_mask")};m={n:v.bool() for n,v in rm.items()};hs={"dataset_file_sha256":file_sha256(ROOT/c["dataset_file"]),"feature_sha256":tensor_sha256(x),"label_sha256":tensor_sha256(y),**{f"{n}_sha256":tensor_sha256(v) for n,v in rm.items()},"raw_graph_edges_sha256":graph_sha256(raw),"training_graph_edges_sha256":graph_sha256(g),"model_py_sha256":file_sha256(ROOT/"methods/gat_v2_gadbench/src/model.py"),"runner_py_sha256":file_sha256(Path(__file__))}
  if any(hs[k]!=v for k,v in c["expected_hashes"].items()) or g.num_edges()!=c["expected_training_graph_edges"]:raise RuntimeError("Frozen T-Finance inputs differ from contract")
  tl=y[m["train_mask"]];weight=[1.,int((tl==0).sum())/int(tl.sum())]
  if weight[1]!=c["expected_anomaly_weight"]:raise RuntimeError("T-Finance class weight mismatch")
  write_json(out/"preflight.json",{"passed":True,"raw_graph":{"nodes":raw.num_nodes(),"edges":raw.num_edges()},"training_graph":{"nodes":g.num_nodes(),"edges":g.num_edges()},"feature_shape":list(x.shape),"mask_counts":{n:int(v.sum()) for n,v in m.items()},"class_weight":weight,"hashes":hs,"edge_access":c["edge_access"]})
  stage="device_transfer";dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");g,x,y=g.to(dev),x.to(dev),y.to(dev);m={n:v.to(dev) for n,v in m.items()};model=GADBenchGATV2(x.shape[1],64,4,0.,2).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);w=torch.tensor(weight,dtype=torch.float32,device=dev);torch.cuda.reset_peak_memory_stats(dev)
  for e in range(1,6):
   stage=f"epoch_{e}";t=time.monotonic();model.train();loss=F.cross_entropy(model(g,x)[m["train_mask"]],y[m["train_mask"]],weight=w);opt.zero_grad(set_to_none=True);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(g,x),dim=1)[:,1].cpu().numpy()
   idx=m["val_mask"].cpu().numpy();vl=y[m["val_mask"]].cpu().numpy();th,vf=best_threshold(vl,p[idx]);v=split_metrics(vl,p[idx],th);r={"epoch":e,"train_loss":float(loss.item()),"validation_f1_macro":vf,"validation_auroc":v["auroc"],"validation_auprc":v["auprc"],"validation_threshold":th,"epoch_wall_time_sec":time.monotonic()-t,"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2};history.append(r);print(json.dumps(r,sort_keys=True),flush=True)
  ti=m["test_mask"].cpu().numpy();ty=y[m["test_mask"]].cpu().numpy();test=split_metrics(ty,p[ti],th);metrics={"status":"smoke","run_type":"smoke","dataset":"tfinance","seed":0,"actual_epochs":5,"stage":"completed","final_test":test,"threshold":th,"wall_time_sec":time.monotonic()-start,"peak_gpu_mb":torch.cuda.max_memory_allocated(dev)/1024**2,"edge_access":c["edge_access"],"hashes":hs}
 except torch.cuda.OutOfMemoryError as e:
  metrics={"status":"OOM","run_type":"smoke","dataset":"tfinance","seed":0,"stage":stage,"error":traceback.format_exc(),"allocated_gpu_mb":torch.cuda.memory_allocated()/1024**2,"reserved_gpu_mb":torch.cuda.memory_reserved()/1024**2,"wall_time_sec":time.monotonic()-start,"edge_access":c["edge_access"]}
 except Exception:
  metrics={"status":"ERROR","run_type":"smoke","dataset":"tfinance","seed":0,"stage":stage,"error":traceback.format_exc(),"wall_time_sec":time.monotonic()-start,"edge_access":c["edge_access"]}
 finally:
  write_json(out/"validation_history.json",history);after=protected_manifest();metrics["protected_manifest_unchanged"]=before==after;write_json(out/"metrics.json",metrics);write_json(out/"protected_manifest_after.json",after);write_json(out/"artifact_sha256s.json",{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()});
  with (out/"runs.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["dataset","seed","run_type","status","stage","actual_epochs","wall_time_sec","peak_gpu_mb","edge_access","error"]);w.writeheader();w.writerow({k:metrics.get(k,"") for k in w.fieldnames})
 print("TFINANCE_SMOKE_COMPLETE "+json.dumps(metrics,sort_keys=True),flush=True)
if __name__=="__main__":main()
