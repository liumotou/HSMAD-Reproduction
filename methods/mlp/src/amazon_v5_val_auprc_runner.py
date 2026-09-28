"""Independent Amazon MLP v5: only validation AUPRC selects the checkpoint/early stop."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from model import FeatureMLP
from utils import file_sha256,setup_seed,tensor_sha256
MASKS={'train_mask':'fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a','val_mask':'2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb','test_mask':'bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1'}
COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6';REF='audit/mlp_reference/GADBench/models/detector.py'
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def th(y,p):
 b=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  v=f1_score(y,(p>t).astype(np.int64),average='macro')
  if v>b[0]:b=(v,float(t))
 return b
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def reference_artifact():
 p=ROOT/REF;lines=p.read_text().splitlines();snippet='\n'.join(f'{i+1}: {lines[i]}' for i in list(range(42,51))+list(range(82,99)))
 got=subprocess.run(['git','-C',str(p.parents[1]),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
 if got!=COMMIT:raise RuntimeError('GADBench commit mismatch')
 return {'repository':'https://github.com/squareroot3/GADBench.git','commit':got,'file':REF,'file_sha256':file_sha256(str(p)),'code_locations':['models/detector.py:43-51','models/detector.py:83-99'],'snippet':snippet}
def evaluate(state,x,y,test,t):
 m=FeatureMLP(25,64,0.).to(x.device);m.load_state_dict(state);m.eval()
 with torch.no_grad():p=torch.softmax(m(x),dim=1)[:,1][test].detach().cpu().numpy()
 labels=y[test].detach().cpu().numpy();pred=(p>t).astype(np.int64)
 return {'f1_macro':float(f1_score(labels,pred,average='macro')),'auroc':float(roc_auc_score(labels,p)),'auprc':float(average_precision_score(labels,p)),'test_predicted_anomaly_count':int(pred.sum()),'test_label_anomaly_count':int(labels.sum()),'probability_min':float(p.min()),'probability_max':float(p.max()),'probability_mean':float(p.mean())}
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',required=True,type=int,choices=range(10));a.add_argument('--mode',required=True,choices=['seed3_verify','formal']);a.add_argument('--config',required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());base=ROOT/'results/experiments/mlp/amazon/protocol_v5_val_auprc_selection';out=base/('seed3_reproduction_check' if z.mode=='seed3_verify' else 'formal')/f'seed_{z.seed}';out.mkdir(parents=True,exist_ok=False);ed=out/'environment';ed.mkdir();r={'method':'MLP','dataset':'amazon','seed':z.seed,'run_type':z.mode,'protocol_version':'v5_val_auprc_selection','status':'ERROR','edge_access':'none','config_sha256':csha(c)}
 try:
  v3=json.loads((ROOT/'methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json').read_text());diff={k:{'v3':v3[k],'v5':c[k]} for k in sorted(set(v3)|set(c)) if v3.get(k)!=c.get(k)}
  if diff!={'selection_metric':{'v3':'validation F1-Macro','v5':'validation AUPRC'}}:raise RuntimeError('v3-v5 config diff not limited to checkpoint metric '+json.dumps(diff,sort_keys=True))
  setup_seed(z.seed);gs,_=dgl.load_graphs(str(ROOT/'datasets/amazon'));n=gs[0].ndata;x=n['feature'].float().contiguous();y=n['label'].long().reshape(-1).contiguous();m={k:n[k].bool().reshape(-1).contiguous() for k in MASKS};fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/amazon')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  for k,e in MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('frozen mask mismatch '+k)
  prefix=int(c['amazon_excluded_prefix_nodes']);covered=m['train_mask']|m['val_mask']|m['test_mask'];pc={k:int(v[:prefix].sum()) for k,v in m.items()}
  if any(pc.values()) or int(covered[:prefix].sum())!=0:raise RuntimeError('Amazon excluded-prefix rule violated')
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','config_diff_v3_to_v5_canonical':{'checkpoint_metric':{'v3':'validation_f1_macro','v5':'validation_auprc'}},'raw_config_diff':diff,'gadbench_auprc_selection_reference':reference_artifact(),'input_sha256':fp,'prefix_mask_counts':pc,'environment':env(d),'edge_access':'none'}
  (out/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,out/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(d),y.to(d);m={k:v.to(d) for k,v in m.items()};model=FeatureMLP(25,64,0.).to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);best={'value':-1.,'epoch':None,'threshold':None,'state':None};hist=[];stall=0
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(d)
  start=time.perf_counter()
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=th(vy,p[vm]);va=float(roc_auc_score(vy,p[vm]));vp=float(average_precision_score(vy,p[vm]));hist.append({'epoch':ep,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'val_auroc':va,'val_auprc':vp,'threshold':t})
   if vp>best['value']:
    best={'value':vp,'epoch':ep,'threshold':t,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}};stall=0
   else:stall+=1
   if stall>=100:break
  (out/'validation_history.json').write_text(json.dumps(hist,indent=2));result=evaluate(best['state'],x,y,m['test_mask'],best['threshold']);ck=out/'best_checkpoint_val_auprc.pt';torch.save({'model_state_dict':best['state'],'seed':z.seed,'selection_metric':'validation_auprc','best_epoch':best['epoch'],'threshold':best['threshold'],'config_sha256':r['config_sha256']},ck)
  r.update({'status':'OK','validation_auprc':best['value'],'best_epoch':best['epoch'],'threshold':best['threshold'],'checkpoint_path':str(ck),'checkpoint_sha256':file_sha256(str(ck)),**result,'wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2) if torch.cuda.is_available() else 0.,**fp})
  if z.mode=='seed3_verify':
   check={'matched':best['epoch']==102 and abs(result['auroc']-.9467263281671678)<1e-12,'expected':{'best_epoch':102,'test_auroc':.9467263281671678},'actual':{'best_epoch':best['epoch'],'test_auroc':result['auroc']}};(out/'reproduction_check.json').write_text(json.dumps(check,indent=2));r['reproduction_check']=check
   if not check['matched']:raise RuntimeError('seed3 AUPRC-best anchor mismatch')
 except Exception:r['error']=traceback.format_exc()
 (out/'metrics.json').write_text(json.dumps(r,indent=2));(out/'result.csv').write_text(','.join(r.keys())+'\n'+','.join(json.dumps(v) if isinstance(v,(dict,list)) else str(v) for v in r.values())+'\n');print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
