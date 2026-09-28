"""One isolated observational checkpoint-selection probe for Amazon v3 seed 3."""
import argparse, hashlib, json, shutil, subprocess, sys, time, traceback
from pathlib import Path
import dgl, numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from model import FeatureMLP
from utils import file_sha256,setup_seed,tensor_sha256
OUT=ROOT/'results/experiments/mlp/amazon/protocol_v3_dropout0/checkpoint_selection_probe/seed_3'
MASKS={'train_mask':'fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a','val_mask':'2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb','test_mask':'bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1'}
BASE={'best_epoch':9,'f1_macro':0.9160324971792786,'auroc':0.8959976675148431}
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def th(y,p):
 b=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  v=f1_score(y,(p>t).astype(np.int64),average='macro')
  if v>b[0]:b=(v,float(t))
 return b
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def evaluate(state,x,y,test,threshold):
 model=FeatureMLP(25,64,0.).to(x.device);model.load_state_dict(state);model.eval()
 with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1][test].detach().cpu().numpy()
 labels=y[test].detach().cpu().numpy();pred=(p>threshold).astype(np.int64)
 return {'f1_macro':float(f1_score(labels,pred,average='macro')),'auroc':float(roc_auc_score(labels,p)),'auprc':float(average_precision_score(labels,p)),'test_predicted_anomaly_count':int(pred.sum()),'test_label_anomaly_count':int(labels.sum()),'probability_min':float(p.min()),'probability_max':float(p.max()),'probability_mean':float(p.mean())}
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());OUT.mkdir(parents=True,exist_ok=False);ed=OUT/'environment';ed.mkdir();record={'method':'MLP','dataset':'amazon','seed':3,'run_type':'checkpoint_selection_probe','protocol_version':'v3_dropout0','status':'ERROR','edge_access':'none','config_sha256':csha(c)}
 try:
  setup_seed(3);gs,_=dgl.load_graphs(str(ROOT/'datasets/amazon'));n=gs[0].ndata;x=n['feature'].float().contiguous();y=n['label'].long().reshape(-1).contiguous();m={k:n[k].bool().reshape(-1).contiguous() for k in MASKS};fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/amazon')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  for k,e in MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('frozen mask mismatch '+k)
  original=ROOT/'results/experiments/mlp/amazon/protocol_v3_dropout0/formal/seed_3/metrics.json';old=json.loads(original.read_text())
  if any(abs(float(old[k])-v)>1e-15 for k,v in BASE.items() if k!='best_epoch') or int(old['best_epoch'])!=BASE['best_epoch']:raise RuntimeError('existing seed3 baseline changed')
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','input_sha256':fp,'baseline':BASE,'baseline_metrics_sha256':file_sha256(str(original)),'environment':env(d),'edge_access':'none','observation_only':'validation F1/AUROC/AUPRC plus three checkpoint snapshots'}
  (OUT/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,OUT/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(d),y.to(d);m={k:v.to(d) for k,v in m.items()};model=FeatureMLP(25,64,0.).to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);best={key:{'value':-1.,'epoch':None,'threshold':None,'state':None} for key in ['f1_macro','auroc','auprc']};hist=[];stall=0
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(d)
  start=time.perf_counter()
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=th(vy,p[vm]);va=float(roc_auc_score(vy,p[vm]));vp=float(average_precision_score(vy,p[vm]));hist.append({'epoch':ep,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'val_auroc':va,'val_auprc':vp,'threshold':t})
   values={'f1_macro':vf,'auroc':va,'auprc':vp}
   for key,value in values.items():
    if value>best[key]['value']:
     best[key]={'value':value,'epoch':ep,'threshold':t,'state':{name:value.detach().cpu().clone() for name,value in model.state_dict().items()}}
   if vf>best['f1_macro']['value']-1e-15 and best['f1_macro']['epoch']==ep:stall=0
   else:stall+=1
   if stall>=100:break
  (OUT/'validation_history.json').write_text(json.dumps(hist,indent=2))
  f1_result=evaluate(best['f1_macro']['state'],x,y,m['test_mask'],best['f1_macro']['threshold'])
  reproduction={'matched':best['f1_macro']['epoch']==BASE['best_epoch'] and abs(f1_result['f1_macro']-BASE['f1_macro'])<1e-12 and abs(f1_result['auroc']-BASE['auroc'])<1e-12,'expected':BASE,'actual':{'best_epoch':best['f1_macro']['epoch'],'f1_macro':f1_result['f1_macro'],'auroc':f1_result['auroc']}}
  (OUT/'reproduction_check.json').write_text(json.dumps(reproduction,indent=2))
  if not reproduction['matched']:raise RuntimeError('F1-best reproduction mismatch; selection comparison intentionally not performed')
  results={}
  for key in ['f1_macro','auroc','auprc']:
   path=OUT/f'checkpoint_val_{key}_best.pt';torch.save({'model_state_dict':best[key]['state'],'seed':3,'selection_metric':'validation_'+key,'best_epoch':best[key]['epoch'],'threshold':best[key]['threshold'],'config_sha256':record['config_sha256']},path);results[key]={'validation_value':best[key]['value'],'best_epoch':best[key]['epoch'],'threshold':best[key]['threshold'],'checkpoint_path':str(path),'checkpoint_sha256':file_sha256(str(path)),**evaluate(best[key]['state'],x,y,m['test_mask'],best[key]['threshold'])}
  record.update({'status':'OK','wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2) if torch.cuda.is_available() else 0.,'reproduction_check':reproduction,'selection_results':results,**fp})
 except Exception:record['error']=traceback.format_exc()
 (OUT/'metrics.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,sort_keys=True))
 if record['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
