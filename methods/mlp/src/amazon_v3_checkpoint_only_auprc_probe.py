"""Amazon seed-3 diagnostic: preserve v3 F1 early stopping; additionally retain AUPRC-best state."""
import argparse,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
OUT=ROOT/'results/experiments/mlp/amazon/v3_checkpoint_only_auprc_probe/seed_3'
MASKS={'train_mask':'fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a','val_mask':'2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb','test_mask':'bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1'}
V3={'best_epoch':9,'f1_macro':.9160324971792786,'auroc':.8959976675148431};AUPRC={'best_epoch':102,'auroc':.9467263281671678}
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def th(y,p):
 b=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  v=f1_score(y,(p>t).astype(np.int64),average='macro')
  if v>b[0]:b=(v,float(t))
 return b
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def load_prob(state,x):
 model=FeatureMLP(25,64,0.).to(x.device);model.load_state_dict(state);model.eval()
 with torch.no_grad():return torch.softmax(model(x),dim=1)[:,1].detach().cpu().numpy()
def test_eval(state,x,y,m):
 p=load_prob(state,x);vy=y[m['val_mask']].detach().cpu().numpy();vf,t=th(vy,p[m['val_mask'].detach().cpu().numpy()]);tp=p[m['test_mask'].detach().cpu().numpy()];ty=y[m['test_mask']].detach().cpu().numpy();pred=(tp>t).astype(np.int64)
 return {'validation_f1_macro_recomputed':vf,'threshold_recomputed_on_validation':t,'f1_macro':float(f1_score(ty,pred,average='macro')),'auroc':float(roc_auc_score(ty,tp)),'auprc':float(average_precision_score(ty,tp)),'test_predicted_anomaly_count':int(pred.sum()),'test_label_anomaly_count':int(ty.sum()),'probability_min':float(tp.min()),'probability_max':float(tp.max()),'probability_mean':float(tp.mean())}
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());OUT.mkdir(parents=True,exist_ok=False);ed=OUT/'environment';ed.mkdir();r={'method':'MLP','dataset':'amazon','seed':3,'run_type':'v3_checkpoint_only_auprc_probe','protocol_version':'v3_checkpoint_only_auprc_probe','status':'ERROR','edge_access':'none','config_sha256':csha(c)}
 try:
  setup_seed(3);gs,_=dgl.load_graphs(str(ROOT/'datasets/amazon'));n=gs[0].ndata;x=n['feature'].float().contiguous();y=n['label'].long().reshape(-1).contiguous();m={k:n[k].bool().reshape(-1).contiguous() for k in MASKS};fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/amazon')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  for k,e in MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('frozen mask mismatch '+k)
  prefix={k:int(v[:3305].sum()) for k,v in m.items()}
  if any(prefix.values()):raise RuntimeError('Amazon excluded prefix leaked')
  diff={'config_value_changes':{},'observability_additions':['validation_auroc recorded every validation epoch','validation_auprc recorded every validation epoch','validation_auprc_best checkpoint saved'],'early_stop_metric':'validation F1-Macro (identical to v3)','checkpoint_selection_metric':'validation AUPRC (observation-only; does not affect early stop)'}
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','config_diff_from_v3':diff,'input_sha256':fp,'prefix_mask_counts':prefix,'environment':env(d),'edge_access':'none'}
  (OUT/'config_diff.json').write_text(json.dumps(diff,indent=2));(OUT/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,OUT/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(d),y.to(d);m={k:v.to(d) for k,v in m.items()};model=FeatureMLP(25,64,0.).to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);f1best={'value':-1.,'epoch':None,'state':None};apbest={'value':-1.,'epoch':None,'state':None};hist=[];stall=0
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(d)
  start=time.perf_counter()
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=th(vy,p[vm]);va=float(roc_auc_score(vy,p[vm]));ap=float(average_precision_score(vy,p[vm]));hist.append({'epoch':ep,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'val_auroc':va,'val_auprc':ap,'threshold':t})
   if vf>f1best['value']:
    f1best={'value':vf,'epoch':ep,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}};stall=0
   else:stall+=1
   if ap>apbest['value']:apbest={'value':ap,'epoch':ep,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
   if stall>=100:break
  (OUT/'validation_history.json').write_text(json.dumps(hist,indent=2));f1path=OUT/'checkpoint_val_f1_macro_best.pt';appath=OUT/'checkpoint_val_auprc_best.pt';torch.save({'model_state_dict':f1best['state'],'seed':3,'selection_metric':'validation_f1_macro','best_epoch':f1best['epoch'],'config_sha256':r['config_sha256']},f1path);torch.save({'model_state_dict':apbest['state'],'seed':3,'selection_metric':'validation_auprc','best_epoch':apbest['epoch'],'config_sha256':r['config_sha256']},appath)
  f1result=test_eval(f1best['state'],x,y,m);f1check={'matched':f1best['epoch']==V3['best_epoch'] and abs(f1result['f1_macro']-V3['f1_macro'])<1e-12 and abs(f1result['auroc']-V3['auroc'])<1e-12,'expected':V3,'actual':{'best_epoch':f1best['epoch'],'f1_macro':f1result['f1_macro'],'auroc':f1result['auroc']}}
  (OUT/'f1_reproduction_check.json').write_text(json.dumps(f1check,indent=2))
  if not f1check['matched']:raise RuntimeError('F1-best v3 reproduction mismatch; AUPRC branch intentionally not compared')
  apresult=test_eval(apbest['state'],x,y,m);apcheck={'matched':apbest['epoch']==AUPRC['best_epoch'] and abs(apresult['auroc']-AUPRC['auroc'])<1e-12,'expected':AUPRC,'actual':{'best_epoch':apbest['epoch'],'auroc':apresult['auroc']}}
  (OUT/'auprc_reproduction_check.json').write_text(json.dumps(apcheck,indent=2))
  if not apcheck['matched']:raise RuntimeError('AUPRC-best probe reproduction mismatch')
  r.update({'status':'OK','total_epochs':len(hist),'early_stop_metric':'validation F1-Macro','checkpoint_selection_metric':'validation AUPRC','f1_best':{'validation_value':f1best['value'],'best_epoch':f1best['epoch'],'checkpoint_path':str(f1path),'checkpoint_sha256':file_sha256(str(f1path)),**f1result},'auprc_best':{'validation_value':apbest['value'],'best_epoch':apbest['epoch'],'checkpoint_path':str(appath),'checkpoint_sha256':file_sha256(str(appath)),**apresult},'wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2) if torch.cuda.is_available() else 0.,**fp})
 except Exception:r['error']=traceback.format_exc()
 (OUT/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
