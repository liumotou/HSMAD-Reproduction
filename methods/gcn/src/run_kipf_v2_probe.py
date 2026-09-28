import argparse,csv,hashlib,json,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from utils import file_sha256,setup_seed,tensor_sha256
from kipf_two_layer import KipfTwoLayerGCN
def jsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def best(y,p,ts):
 r=(-1.,ts[0])
 for t in ts:
  z=f1_score(y,(p>t).astype(np.int64),average='macro')
  if z>r[0]:r=(float(z),float(t))
 return r
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();cp=Path(a.config);c=json.loads(cp.read_text());ds=c['dataset'];out=ROOT/'results/experiments/gcn'/ds/c['protocol_version']/c['run_type']/'seed_0';out.mkdir(parents=True,exist_ok=False);(out/'environment').mkdir();r={'method':'GCN','protocol_version':c['protocol_version'],'dataset':ds,'seed':0,'run_type':c['run_type'],'status':'ERROR','edge_access':c['edge_access'],'config_sha256':jsha(c),'code_sha256':file_sha256(str(Path(__file__))),'model_sha256':file_sha256(str(ROOT/'methods/gcn/src/kipf_two_layer.py'))}
 try:
  setup_seed(0);raw=dgl.load_graphs(str(ROOT/'datasets'/ds))[0][0];req=['feature','label','train_mask','val_mask','test_mask'];x=raw.ndata['feature'].float().contiguous();y=raw.ndata['label'].long().reshape(-1).contiguous();m={k:raw.ndata[k].bool().reshape(-1).contiguous() for k in req[2:]}
  if x.shape[1]!=c['input_dim']:raise RuntimeError('input dimension mismatch')
  if ds=='amazon' and any(int(v[:3305].sum()) for v in m.values()):raise RuntimeError('Amazon prefix covered')
  fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets'/ds)),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));g.ndata['feature']=x;g.ndata['label']=y;d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');g=g.to(d);y=y.to(d);m={k:v.to(d) for k,v in m.items()};snap=dict(c);pre={'input_sha256':fp,'raw_graph':{'nodes':raw.num_nodes(),'edges':raw.num_edges()},'training_graph':{'nodes':g.num_nodes(),'edges':g.num_edges(),'preprocess':c['graph_preprocess']},'mask_counts':{k:int(v.sum().cpu()) for k,v in m.items()},'edge_access':c['edge_access'],'config':snap};(out/'preflight.json').write_text(json.dumps(pre,indent=2));(out/'config_snapshot.json').write_text(json.dumps(snap,indent=2));(out/'environment/framework_versions.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'dgl':dgl.__version__,'gpu':torch.cuda.get_device_name(d)},indent=2))
  model=KipfTwoLayerGCN(c['input_dim'],64,2,0.).to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);fb={'v':-1,'s':None,'e':0};ab={'v':-1,'s':None,'e':0};stall=0;h=[];torch.cuda.reset_peak_memory_stats(d);st=time.perf_counter()
  for e in range(1,201):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(g)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(g),1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=best(vy,p[vm],c['threshold_candidates']);va=float(roc_auc_score(vy,p[vm]));au=float(average_precision_score(vy,p[vm]));h.append({'epoch':e,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'val_auroc':va,'val_auprc':au,'threshold':t})
   if vf>fb['v']:fb={'v':vf,'s':state(model),'e':e};stall=0
   else:stall+=1
   if au>ab['v']:ab={'v':au,'s':state(model),'e':e}
   print(json.dumps({'epoch':e,'val_f1_macro':vf,'val_auroc':va,'val_auprc':au,'f1_patience_count':stall}),flush=True)
   if stall>50:break
  fc,ac=out/'checkpoint_val_f1_best.pt',out/'checkpoint_val_auprc_best.pt';torch.save({'model_state_dict':fb['s'],'epoch':fb['e']},fc);torch.save({'model_state_dict':ab['s'],'epoch':ab['e']},ac);model.load_state_dict(ab['s']);model.eval()
  with torch.no_grad():p=torch.softmax(model(g),1)[:,1].cpu().numpy()
  vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();_,t=best(vy,p[vm],c['threshold_candidates']);tm=m['test_mask'].cpu().numpy();ty=y[m['test_mask']].cpu().numpy();tp=p[tm];r.update({'status':c['run_type'],'f1_macro':float(f1_score(ty,(tp>t).astype(np.int64),average='macro')),'auroc':float(roc_auc_score(ty,tp)),'threshold':t,'best_epoch':ab['e'],'early_stop_best_epoch':fb['e'],'validation_auprc':ab['v'],'epochs_executed':len(h),'wall_time_sec':time.perf_counter()-st,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2),'checkpoint_sha256':file_sha256(str(ac)),**fp,'raw_nodes':raw.num_nodes(),'raw_edges':raw.num_edges(),'training_nodes':g.num_nodes(),'training_edges':g.num_edges()});(out/'validation_history.json').write_text(json.dumps(h,indent=2))
 except Exception:r['error']=traceback.format_exc()
 (out/'metrics.json').write_text(json.dumps(r,indent=2));
 with (out/'runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
 print(json.dumps(r));
 if r['status'] not in ['smoke','diagnostic']:raise SystemExit(1)
if __name__=='__main__':main()
