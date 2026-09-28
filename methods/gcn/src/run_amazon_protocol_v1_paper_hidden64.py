import argparse,csv,hashlib,json,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from utils import file_sha256,setup_seed,tensor_sha256
sys.path.append(str(ROOT/'audit/mlp_reference/GADBench'));from models.gnn import GCN
CFG=ROOT/'methods/gcn/configs/amazon_protocol_v1_paper_hidden64_formal.json';OUTROOT=ROOT/'results/experiments/gcn/amazon/protocol_v1_paper_hidden64/formal';REF=ROOT/'audit/mlp_reference/GADBench/models/gnn.py';COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6'
def jsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def best(y,p,ts):
 r=(-1.,ts[0])
 for t in ts:
  z=f1_score(y,(p>t).astype(np.int64),average='macro')
  if z>r[0]:r=(float(z),float(t))
 return r
def clone(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def csvone(p,r):
 with p.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(r));w.writeheader();w.writerow(r)
def run(seed):
 c=json.loads(CFG.read_text());out=OUTROOT/f'seed_{seed}';out.mkdir(parents=True,exist_ok=False);(out/'environment').mkdir()
 r={'method':'GCN','protocol_version':c['protocol_version'],'dataset':'amazon','seed':seed,'run_type':'formal','status':'ERROR','edge_access':c['edge_access'],'config_sha256':jsha(c),'code_sha256':file_sha256(str(Path(__file__)))}
 try:
  if seed not in c['training_seeds']:raise ValueError('unfrozen seed')
  setup_seed(seed);raw=dgl.load_graphs(str(ROOT/'datasets/amazon'))[0][0];req=['feature','label','train_mask','val_mask','test_mask']
  if any(k not in raw.ndata for k in req):raise RuntimeError('missing frozen Amazon ndata')
  x=raw.ndata['feature'].float().contiguous();y=raw.ndata['label'].long().reshape(-1).contiguous();m={k:raw.ndata[k].bool().reshape(-1).contiguous() for k in req[2:]};prefix=c['exclude_prefix_nodes_from_masks_and_metrics']
  if any(int(v[:prefix].sum()) for v in m.values()):raise RuntimeError('first 3305 nodes covered by mask')
  if int(((m['train_mask'].int()+m['val_mask'].int()+m['test_mask'].int())>1).sum()):raise RuntimeError('mask overlap')
  fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/amazon')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));g.ndata['feature']=x;g.ndata['label']=y;d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');g=g.to(d);y=y.to(d);m={k:v.to(d) for k,v in m.items()}
  snap=dict(c);snap['active_seed']=seed;stats={k:{'nodes':int(v.sum().cpu()),'anomalies':int(y[v].sum().cpu())} for k,v in m.items()};pre={'reference_repo':'https://github.com/squareRoot3/GADBench.git','reference_commit':COMMIT,'reference_file_sha256':file_sha256(str(REF)),'runner_code_sha256':r['code_sha256'],'input_sha256':fp,'raw_graph':{'nodes':raw.num_nodes(),'edges':raw.num_edges()},'training_graph':{'nodes':g.num_nodes(),'edges':g.num_edges(),'preprocess':c['graph_preprocess']},'frozen_mask_stats':stats,'uncovered_prefix':{'node_count':prefix,'covered_node_count':0},'edge_access':c['edge_access'],'config':snap};(out/'preflight.json').write_text(json.dumps(pre,indent=2));(out/'config_snapshot.json').write_text(json.dumps(snap,indent=2));(out/'environment/framework_versions.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'dgl':dgl.__version__,'gpu':torch.cuda.get_device_name(d)},indent=2))
  model=GCN(x.shape[1],64,2,2,1,0.,'ReLU').to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);fb={'value':-1,'state':None,'epoch':0};ab={'value':-1,'state':None,'epoch':0};stall=0;hist=[];torch.cuda.reset_peak_memory_stats(d);start=time.perf_counter()
  for e in range(1,c['max_epoch']+1):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(g)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(g),1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=best(vy,p[vm],c['threshold_candidates']);va=float(roc_auc_score(vy,p[vm]));ap=float(average_precision_score(vy,p[vm]));hist.append({'epoch':e,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'val_auroc':va,'val_auprc':ap,'threshold':t})
   if vf>fb['value']:fb={'value':vf,'state':clone(model),'epoch':e};stall=0
   else:stall+=1
   if ap>ab['value']:ab={'value':ap,'state':clone(model),'epoch':e}
   print(json.dumps({'seed':seed,'epoch':e,'val_f1_macro':vf,'val_auroc':va,'val_auprc':ap,'f1_patience_count':stall}),flush=True)
   if stall>c['patience']:break
  fc,ac=out/'checkpoint_val_f1_earlystop_best.pt',out/'checkpoint_val_auprc_best.pt';torch.save({'model_state_dict':fb['state'],'epoch':fb['epoch']},fc);torch.save({'model_state_dict':ab['state'],'epoch':ab['epoch']},ac);model.load_state_dict(ab['state']);model.eval()
  with torch.no_grad():p=torch.softmax(model(g),1)[:,1].cpu().numpy()
  vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();_,t=best(vy,p[vm],c['threshold_candidates']);tm=m['test_mask'].cpu().numpy();ty=y[m['test_mask']].cpu().numpy();tp=p[tm]
  r.update({'status':'OK','f1_macro':float(f1_score(ty,(tp>t).astype(np.int64),average='macro')),'auroc':float(roc_auc_score(ty,tp)),'threshold':t,'best_epoch':ab['epoch'],'validation_auprc':ab['value'],'early_stop_best_epoch':fb['epoch'],'epochs_executed':len(hist),'wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2),'checkpoint_sha256':file_sha256(str(ac)),'f1_checkpoint_sha256':file_sha256(str(fc)),**fp,'raw_nodes':raw.num_nodes(),'raw_edges':raw.num_edges(),'training_nodes':g.num_nodes(),'training_edges':g.num_edges(),'uncovered_prefix_nodes':prefix});(out/'validation_history.json').write_text(json.dumps(hist,indent=2))
 except Exception:r['error']=traceback.format_exc()
 (out/'metrics.json').write_text(json.dumps(r,indent=2));csvone(out/'runs.csv',r);print(json.dumps(r),flush=True)
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);run(a.parse_args().seed)
