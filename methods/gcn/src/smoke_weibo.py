import hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from train import EXPECTED_WEIBO_MASKS,load_feature_data
from utils import file_sha256,setup_seed,tensor_sha256
sys.path.append(str(ROOT/'audit/mlp_reference/GADBench'))
from models.gnn import GCN
OUT=ROOT/'results/experiments/gcn/weibo/smoke/seed_0'
REF=ROOT/'audit/mlp_reference/GADBench/models/gnn.py'; COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6'
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def th(y,p):
 b=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  z=f1_score(y,(p>t).astype(np.int64),average='macro')
  if z>b[0]:b=(z,float(t))
 return b
def main():
 c=json.loads((ROOT/'methods/gcn/configs/weibo_smoke_seed0.json').read_text());OUT.mkdir(parents=True,exist_ok=False);(OUT/'environment').mkdir();r={'method':'GCN','dataset':'weibo','seed':0,'run_type':'smoke','status':'ERROR','edge_access':'graph_edges_required','config_sha256':csha(c)}
 try:
  setup_seed(0); raw=dgl.load_graphs(str(ROOT/'datasets/weibo'))[0][0]; x,y,m=load_feature_data(ROOT/'datasets/weibo')
  fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/weibo')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  if any(fp[k+'_sha256']!=v for k,v in EXPECTED_WEIBO_MASKS.items()):raise RuntimeError('frozen Weibo mask mismatch')
  g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw))); g.ndata['feature']=x; g.ndata['label']=y
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); g=g.to(d);y=y.to(d);m={k:v.to(d) for k,v in m.items()}
  pre={'reference_repo':'https://github.com/squareRoot3/GADBench.git','reference_commit':COMMIT,'reference_file_sha256':file_sha256(str(REF)),'input_sha256':fp,'raw_graph':{'nodes':raw.num_nodes(),'edges':raw.num_edges()},'training_graph':{'nodes':g.num_nodes(),'edges':g.num_edges(),'preprocess':c['graph_preprocess']},'edge_access':'graph_edges_required','config':c};(OUT/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(ROOT/'methods/gcn/configs/weibo_smoke_seed0.json',OUT/'config_snapshot.json');(OUT/'environment/framework_versions.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'dgl':dgl.__version__,'gpu':torch.cuda.get_device_name(d)},indent=2))
  model=GCN(x.shape[1],32,2,2,1,0.,'ReLU').to(d);opt=torch.optim.Adam(model.parameters(),lr=.01);hist=[];best={'value':-1,'state':None,'epoch':0}
  torch.cuda.reset_peak_memory_stats(d);start=time.perf_counter()
  for e in range(1,6):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(g)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(g),1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=th(vy,p[vm]);ap=float(average_precision_score(vy,p[vm]));hist.append({'epoch':e,'loss':float(loss.cpu()),'val_f1_macro':vf,'val_auprc':ap,'threshold':t})
   if ap>best['value']:best={'value':ap,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':e}
  ck=OUT/'checkpoint_val_auprc_best.pt';torch.save({'model_state_dict':best['state'],'epoch':best['epoch']},ck);model.load_state_dict(best['state']);model.eval()
  with torch.no_grad():p=torch.softmax(model(g),1)[:,1].cpu().numpy()
  vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=th(vy,p[vm]);tm=m['test_mask'].cpu().numpy();ty=y[m['test_mask'].cpu()].cpu().numpy() if False else y[m['test_mask']].cpu().numpy();tp=p[tm]
  r.update({'status':'smoke','f1_macro':float(f1_score(ty,(tp>t).astype(np.int64),average='macro')),'auroc':float(roc_auc_score(ty,tp)),'threshold':t,'best_epoch':best['epoch'],'validation_auprc':best['value'],'wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2),'checkpoint_sha256':file_sha256(str(ck)),**fp,'raw_nodes':raw.num_nodes(),'raw_edges':raw.num_edges(),'training_nodes':g.num_nodes(),'training_edges':g.num_edges()})
  (OUT/'validation_history.json').write_text(json.dumps(hist,indent=2))
 except Exception:r['error']=traceback.format_exc()
 (OUT/'metrics.json').write_text(json.dumps(r,indent=2));(OUT/'runs.csv').write_text(','.join(r)+'\n'+','.join(json.dumps(v) if isinstance(v,(dict,list)) else str(v) for v in r.values())+'\n');print(json.dumps(r))
 if r['status']!='smoke':raise SystemExit(1)
if __name__=='__main__':main()
