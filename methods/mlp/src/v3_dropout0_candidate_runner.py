"""Independent Weibo v3 dropout-zero candidate runner; feature-only MLP."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.train import EXPECTED_WEIBO_MASKS,best_validation_threshold,load_feature_data
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
OUT=ROOT/'results/experiments/mlp/weibo/protocol_v3_dropout0';COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6';REFSHA='6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d'
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def codehash():return {f:file_sha256(str(ROOT/f)) for f in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def append(row):
 p=OUT/'runs.csv';exists=p.exists()
 with p.open('a',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(row));(w.writeheader() if not exists else None);w.writerow(row)
def main():
 q=argparse.ArgumentParser();q.add_argument('--seed',type=int,required=True,choices=range(10));q.add_argument('--config',required=True);a=q.parse_args();cp=Path(a.config).resolve();c=json.loads(cp.read_text());run=OUT/f'seed_{a.seed}';run.mkdir(parents=True,exist_ok=False);ed=run/'environment';ed.mkdir();r={'method':'MLP','dataset':'weibo','seed':a.seed,'run_type':'candidate','status':'ERROR','protocol_version':'v3_dropout0_candidate','edge_access':'none','config_sha256':csha(c)}
 try:
  setup_seed(a.seed);ds=ROOT/'datasets/weibo';x,y,m=load_feature_data(ds);fp={'dataset_file_sha256':file_sha256(str(ds)),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  for k,e in EXPECTED_WEIBO_MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('frozen mask mismatch '+k)
  ref=ROOT/'audit/mlp_reference/GADBench/models/gnn.py'
  if file_sha256(str(ref))!=REFSHA:raise RuntimeError('reference SHA mismatch')
  got=subprocess.run(['git','-C',str(ref.parents[1]),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
  if got!=COMMIT:raise RuntimeError('reference commit mismatch')
  if c['dropout']!=0. or c['class_weight'] is not None:raise RuntimeError('candidate config violates dropout0/no-weight')
  dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','reference_commit':got,'reference_file_sha256':REFSHA,'input_sha256':fp,'code_sha256':codehash(),'environment':env(dev),'edge_access':'none','class_weight':None,'dropout':0.0}
  (run/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,run/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(dev),y.to(dev);m={k:v.to(dev) for k,v in m.items()};model=FeatureMLP(x.shape[1],64,0.).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);best=None;state=None;stall=0;start=time.perf_counter()
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(dev)
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():pr=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vf,t=best_validation_threshold(y[m['val_mask']].cpu().numpy(),pr[vm])
   if best is None or vf>best['val_f1']:
    tm=m['test_mask'].cpu().numpy();tl=y[m['test_mask']].cpu().numpy();tp=pr[tm];pd=(tp>t).astype(np.int64);best={'val_f1':vf,'threshold':t,'best_epoch':ep,'f1_macro':float(f1_score(tl,pd,average='macro')),'auroc':float(roc_auc_score(tl,np.nan_to_num(tp)))};state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stall=0
   else:stall+=1
   if stall>=100:break
  torch.save({'model_state_dict':state,'seed':a.seed,'best_epoch':best['best_epoch'],'threshold':best['threshold'],'config_sha256':r['config_sha256']},run/'best_checkpoint.pt');r.update(best);r.update(fp);r.update({'status':'OK','wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(dev)/1024**2) if torch.cuda.is_available() else 0.,'parameter_count':sum(z.numel() for z in model.parameters()),'class_weight':None,'dropout':0.0,'reference_commit':got,'reference_file_sha256':REFSHA,'code_sha256':json.dumps(codehash(),sort_keys=True),'auroc_score':'softmax(logits)[:,1]'})
 except Exception:r['error']=traceback.format_exc()
 (run/'metrics.json').write_text(json.dumps(r,indent=2));append(r);print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
