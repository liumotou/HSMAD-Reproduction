"""One isolated T-Social MLP v5a diagnostic; feature-only with frozen 1-D masks."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
OUT=ROOT/os.environ.get('MLP_TFINANCE_V5A_FORMAL_ROOT','results/experiments/mlp/tsocial/protocol_v5a_f1_earlystop_auprc_checkpoint/formal')
MASKS=('train_mask','val_mask','test_mask')
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def ch():return {p:file_sha256(str(ROOT/p)) for p in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
def th(y,p):
 b=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  f=f1_score(y,(p>t).astype(np.int64),average='macro')
  if f>b[0]:b=(f,float(t))
 return b
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def score(state,x,y,m):
 model=FeatureMLP(10,64,0.).to(x.device);model.load_state_dict(state);model.eval()
 with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1].detach().cpu().numpy()
 vm=m['val_mask'].detach().cpu().numpy();vy=y[m['val_mask']].detach().cpu().numpy();vf,t=th(vy,p[vm]);tm=m['test_mask'].detach().cpu().numpy();ty=y[m['test_mask']].detach().cpu().numpy();tp=p[tm];pred=(tp>t).astype(np.int64)
 return {'validation_f1_macro_recomputed':vf,'threshold_recomputed_on_validation':t,'f1_macro':float(f1_score(ty,pred,average='macro')),'auroc':float(roc_auc_score(ty,tp)),'auprc':float(average_precision_score(ty,tp)),'test_predicted_anomaly_count':int(pred.sum()),'test_label_anomaly_count':int(ty.sum()),'probability_min':float(tp.min()),'probability_max':float(tp.max()),'probability_mean':float(tp.mean())}
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',required=True,type=int,choices=range(10));a.add_argument('--config',required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());run=OUT/f'seed_{z.seed}';run.mkdir(parents=True,exist_ok=False);ed=run/'environment';ed.mkdir();r={'method':'MLP','dataset':'tsocial','seed':z.seed,'run_type':'formal','protocol_version':'v5a_f1_earlystop_auprc_checkpoint','status':'ERROR','edge_access':'none','config_sha256':csha(c),'source_reference':c['source_reference'],'execution_protocol':c['execution_protocol'],'not_claimed_as':c['not_claimed_as']}
 try:
  setup_seed(z.seed);gs,_=dgl.load_graphs(str(ROOT/'datasets/tsocial'));n=gs[0].ndata;x=n['feature'].float().contiguous();y=n['label'].long().reshape(-1).contiguous();m={k:n[k].bool().reshape(-1).contiguous() for k in MASKS};fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/tsocial')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()},'nodes':int(y.numel()),'feature_shape':list(x.shape)}
  for k in MASKS:
   if m[k].numel()!=y.numel():raise RuntimeError('invalid frozen mask '+k)
  if x.shape!=(y.numel(),10) or c['input_dim']!=10:raise RuntimeError('unexpected T-Social feature dimension')
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','input_sha256':fp,'frozen_project_masks_match':True,'mask_counts':{k:int(v.sum()) for k,v in m.items()},'ignored_dataset_native_multi_masks':[k for k in ['train_masks','val_masks','test_masks'] if k in n],'code_sha256':ch(),'environment':env(d),'edge_access':'none','source_reference':r['source_reference'],'execution_protocol':r['execution_protocol'],'not_claimed_as':r['not_claimed_as']}
  (run/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,run/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(d),y.to(d);m={k:v.to(d) for k,v in m.items()};model=FeatureMLP(10,64,0.).to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);fb={'value':-1.,'epoch':None,'state':None};ab={'value':-1.,'epoch':None,'state':None};hist=[];stall=0
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(d)
  start=time.perf_counter()
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vf,t=th(vy,p[vm]);va=float(roc_auc_score(vy,p[vm]));ap=float(average_precision_score(vy,p[vm]));hist.append({'epoch':ep,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'val_auroc':va,'val_auprc':ap,'threshold':t})
   if vf>fb['value']:
    fb={'value':vf,'epoch':ep,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}};stall=0
   else:stall+=1
   if ap>ab['value']:ab={'value':ap,'epoch':ep,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
   if stall>=100:break
  (run/'validation_history.json').write_text(json.dumps(hist,indent=2));fpck=run/'checkpoint_val_f1_macro_best.pt';apck=run/'checkpoint_val_auprc_best.pt';torch.save({'model_state_dict':fb['state'],'seed':z.seed,'selection_metric':'validation_f1_macro','best_epoch':fb['epoch'],'config_sha256':r['config_sha256']},fpck);torch.save({'model_state_dict':ab['state'],'seed':z.seed,'selection_metric':'validation_auprc','best_epoch':ab['epoch'],'config_sha256':r['config_sha256']},apck);fscore=score(fb['state'],x,y,m);ascore=score(ab['state'],x,y,m)
  r.update({'status':'OK','early_stop_metric':'validation F1-Macro','checkpoint_selection_metric':'validation AUPRC','total_epochs':len(hist),'validation_auprc':ab['value'],'best_epoch':ab['epoch'],'threshold':ascore['threshold_recomputed_on_validation'],'checkpoint_path':str(apck),'checkpoint_sha256':file_sha256(str(apck)),**ascore,'f1_best_checkpoint':{'validation_value':fb['value'],'best_epoch':fb['epoch'],'checkpoint_path':str(fpck),'checkpoint_sha256':file_sha256(str(fpck)),**fscore},'wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2) if torch.cuda.is_available() else 0.,'code_sha256':json.dumps(ch(),sort_keys=True),**fp})
 except Exception:r['error']=traceback.format_exc()
 (run/'metrics.json').write_text(json.dumps(r,indent=2));(run/'result.csv').write_text(','.join(r.keys())+'\n'+','.join(json.dumps(v) if isinstance(v,(dict,list)) else str(v) for v in r.values())+'\n');print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
