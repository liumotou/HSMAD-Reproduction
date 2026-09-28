"""Independent BWGNN formal-candidate runner; never writes smoke directories."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, time
from pathlib import Path
import dgl, numpy as np, torch
from sklearn.metrics import average_precision_score
from methods.bwgnn.src.model import GADBenchBWGNN
from methods.bwgnn.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics
from methods.bwgnn.src.run_smoke import prepare_training_graph, sha256_file, sha256_tensor
from methods.project_paths import project_root
ROOT = project_root()
def dataset_contract(dataset):
 contracts={'weibo':{'dataset_file':'datasets/weibo','result_dataset':'weibo','expected_nodes':8405},'tolokers':{'dataset_file':'datasets/tolokers','result_dataset':'tolokers','expected_nodes':11758},'amazon':{'dataset_file':'datasets/amazon','result_dataset':'amazon','expected_nodes':11944},'tfinance':{'dataset_file':'datasets/tfinance','result_dataset':'tfinance','expected_nodes':39357}}
 if dataset not in contracts: raise ValueError('unsupported dataset')
 return contracts[dataset]
def formal_contract(): return {'max_epoch':200,'patience':50,'early_stop_metric':'validation_auprc','checkpoint_metric':'validation_auprc','threshold_protocol':'validation_F1_macro_grid_0.05_to_0.95'}
def execution_contract(run_type):
 if run_type=='smoke': return {'max_epoch':5,'patience':5,'run_type':'smoke'}
 if run_type in ('diagnostic','formal'): return dict(formal_contract(),run_type=run_type)
 raise ValueError('unsupported run type')
def setup(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);dgl.seed(seed)
def dump(path,data): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,sort_keys=True))
def run(seed,base,run_type='formal'):
 cfg=dict(base);cfg.update(execution_contract(run_type));cfg.update({'seed':seed,'positioning':'candidate_protocol_not_author_exact'})
 contract=dataset_contract(cfg['dataset'])
 if cfg['dataset_file'] != contract['dataset_file']: raise RuntimeError('dataset/config mismatch')
 output=ROOT/f"results/experiments/bwgnn/{contract['result_dataset']}/bwgnn_gadbench_h64_candidate/{run_type}/seed_{seed}"
 if output.exists(): raise FileExistsError(output)
 setup(seed);raw_path=ROOT/cfg['dataset_file'];raw=dgl.load_graphs(str(raw_path))[0][0];g=prepare_training_graph(raw);x=raw.ndata['feature'].float();y=raw.ndata['label'].long().reshape(-1);m={k:raw.ndata[k].bool() for k in ('train_mask','val_mask','test_mask')}
 hashes={'dataset_file_sha256':sha256_file(raw_path),'feature_sha256':sha256_tensor(x),'label_sha256':sha256_tensor(y),**{k+'_sha256':sha256_tensor(v) for k,v in m.items()},'model_py_sha256':sha256_file(ROOT/'methods/bwgnn/src/model.py'),'protocol_py_sha256':sha256_file(ROOT/'methods/bwgnn/src/protocol.py'),'runner_py_sha256':sha256_file(Path(__file__))}
 output.mkdir(parents=True);dump(output/'config_snapshot.json',cfg);dump(output/'preflight.json',{'passed':True,'nodes':g.num_nodes(),'training_edges':g.num_edges(),'mask_counts':{k:int(v.sum()) for k,v in m.items()},'hashes':hashes})
 device=torch.device('cuda');g,x,y=g.to(device),x.to(device),y.to(device);m={k:v.to(device) for k,v in m.items()};tl=y[m['train_mask']];w=torch.tensor([1.,int((tl==0).sum())/int(tl.sum())],device=device)
 model=GADBenchBWGNN(x.shape[1],64,2,2,2,0.).to(device);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.);torch.cuda.reset_peak_memory_stats(device);start=time.monotonic();best=-1.;bad=0;history=[];best_state=None;best_epoch=0
 for epoch in range(1,cfg['max_epoch']+1):
  model.train();z=model(g);loss=masked_cross_entropy(z,y,m['train_mask'],w);opt.zero_grad(set_to_none=True);loss.backward();opt.step();model.eval()
  with torch.no_grad():p=torch.softmax(model(g),1)[:,1]
  vt,vf=select_validation_threshold(y,p,m['val_mask']);truth=y[m['val_mask']].cpu().numpy();score=p[m['val_mask']].cpu().numpy();va=float(average_precision_score(truth,score));record={'epoch':epoch,'train_loss':float(loss),'validation_auprc':va,'validation_f1_macro':vf,'validation_threshold':vt};history.append(record)
  if va>best: best,bad,best_epoch,best_state=va,0,epoch,{k:v.cpu() for k,v in model.state_dict().items()}
  else: bad+=1
  if bad>=cfg['patience']: break
 model.load_state_dict(best_state);model.eval();
 with torch.no_grad():p=torch.softmax(model(g),1)[:,1]
 threshold,vf=select_validation_threshold(y,p,m['val_mask']);metrics=test_metrics(y,p,m['test_mask'],threshold);metrics.update({'method':'BWGNN-GADBench-h64','protocol_version':'bwgnn_gadbench_h64_candidate','positioning':'candidate_protocol_not_author_exact','dataset':cfg['dataset'],'seed':seed,'run_type':run_type,'status':'OK','actual_epochs':epoch,'best_epoch':best_epoch,'validation_auprc':best,'validation_f1_macro':vf,'threshold':threshold,'wall_time_sec':time.monotonic()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(device)/1024**2),'hashes':hashes})
 torch.save({'epoch':best_epoch,'model_state_dict':best_state,'config':cfg},output/'checkpoint_auprc_best.pt');dump(output/'validation_history.json',history);dump(output/'metrics.json',metrics);dump(output/'artifact_sha256s.json',{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()});return metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seeds',default='0');p.add_argument('--run-type',default='diagnostic');a=p.parse_args();base=json.loads(Path(a.config).read_text());root=ROOT/f"results/experiments/bwgnn/{dataset_contract(base['dataset'])['result_dataset']}/bwgnn_gadbench_h64_candidate"/a.run_type;root.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in map(int,a.seeds.split(',')):
  try: rows.append(run(seed,base,a.run_type))
  except Exception as exc: rows.append({'seed':seed,'status':'ERROR','error':repr(exc),'run_type':a.run_type})
 with (root/'runs.csv').open('w',newline='') as f:
  fields=sorted({k for row in rows for k in row});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(json.dumps(rows,sort_keys=True))
if __name__=='__main__':main()
