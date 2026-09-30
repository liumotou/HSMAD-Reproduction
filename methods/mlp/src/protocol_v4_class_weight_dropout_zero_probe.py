"""One seed v4 probe: train-ratio class weight + zero dropout; feature-only."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.train import EXPECTED_WEIBO_MASKS,best_validation_threshold,load_feature_data
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
P=ROOT/'results/experiments/mlp/weibo/protocol_v4_class_weight_dropout_zero_probe';V1=ROOT/'results/experiments/mlp/weibo/formal/seed_0';COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6';REFSHA='6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d'
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def hashes():return {f:file_sha256(str(ROOT/f)) for f in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def proj(c):return {k:c[k] for k in ['model','dataset','architecture','activation','hidden_dim','dropout','loss','class_weight','sampling','optimizer','learning_rate','weight_decay','max_epoch','patience','selection_metric','threshold_candidates','edge_access']}
def diff(v1,v4):return {k:{'v1':v1[k],'v4':v4[k]} for k in v1 if v1[k]!=v4[k]}
def csvout(path,row):
 with path.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(row));w.writeheader();w.writerow(row)
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,default=0);a.add_argument('--config',required=True);z=a.parse_args()
 if z.seed!=0:raise ValueError('Only seed=0 is authorized')
 cp=Path(z.config).resolve();c=json.loads(cp.read_text());v1c=json.loads((ROOT/'methods/mlp/configs/weibo_formal.json').read_text());v1p=proj(v1c);v4p=proj(c);d=diff(v1p,v4p);expected={'dropout':{'v1':0.5,'v4':0.0},'class_weight':{'v1':None,'v4':c['class_weight']}}
 if d!=expected:raise RuntimeError('v1/v4 diff is not exactly class_weight+dropout: '+json.dumps(d,sort_keys=True))
 run=P/'seed_0';run.mkdir(parents=True,exist_ok=False);ed=run/'environment';ed.mkdir();r={'method':'MLP','dataset':'weibo','seed':0,'run_type':'protocol_v4_class_weight_dropout_zero_probe','status':'ERROR','protocol_version':'v4','edge_access':'none','config_sha256':csha(c)}
 try:
  setup_seed(0);(P/'config_diff_preflight.json').write_text(json.dumps({'status':'PASS','only_semantic_training_differences':['class_weight','dropout'],'diff':d,'v1_class_weight':None,'v4_class_weight_expected':c['class_weight']['expected'],'v1_dropout':0.5,'v4_dropout':0.0},indent=2))
  ds=ROOT/'datasets/weibo';x,y,m=load_feature_data(ds);fp={'dataset_file_sha256':file_sha256(str(ds)),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  for k,e in EXPECTED_WEIBO_MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('mask mismatch '+k)
  v1=json.loads((V1/'metrics.json').read_text())
  for k in fp:
   if fp[k]!=v1[k]:raise RuntimeError('v1 frozen input mismatch '+k)
  ref=ROOT/'audit/mlp_reference/GADBench/models/gnn.py'
  if file_sha256(str(ref))!=REFSHA:raise RuntimeError('reference SHA mismatch')
  got=subprocess.run(['git','-C',str(ref.parents[1]),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
  if got!=COMMIT:raise RuntimeError('reference commit mismatch')
  ty=y[m['train_mask']];normal=int((ty==0).sum());anomaly=int((ty==1).sum());w=[1.0,normal/anomaly]
  if w!=c['class_weight']['expected']:raise RuntimeError('computed class weight mismatch')
  dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','reference_commit':got,'reference_file_sha256':REFSHA,'input_sha256':fp,'code_sha256':hashes(),'environment':env(dev),'v1_v4_config_diff':d,'train_normal_count':normal,'train_anomaly_count':anomaly,'class_weight':w,'edge_access':'none'}
  (run/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,run/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(dev),y.to(dev);m={k:v.to(dev) for k,v in m.items()};cw=torch.tensor(w,dtype=torch.float32,device=dev);model=FeatureMLP(x.shape[1],64,0.0).to(dev);opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);best=None;state=None;stall=0;start=time.perf_counter()
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(dev)
  for ep in range(1,c['max_epoch']+1):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']],weight=cw);loss.backward();opt.step();model.eval()
   with torch.no_grad():pr=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vf,t=best_validation_threshold(y[m['val_mask']].cpu().numpy(),pr[vm])
   if best is None or vf>best['val_f1']:
    tm=m['test_mask'].cpu().numpy();tl=y[m['test_mask']].cpu().numpy();tp=pr[tm];pd=(tp>t).astype(np.int64);best={'val_f1':vf,'threshold':t,'best_epoch':ep,'f1_macro':float(f1_score(tl,pd,average='macro')),'auroc':float(roc_auc_score(tl,np.nan_to_num(tp))),'test_predicted_anomaly_count':int(pd.sum()),'test_predicted_anomaly_ratio':float(pd.mean()),'test_confusion_matrix_labels_0_1':confusion_matrix(tl,pd,labels=[0,1]).tolist()};state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stall=0
   else:stall+=1
   if stall>=c['patience']:break
  torch.save({'model_state_dict':state,'seed':0,'best_epoch':best['best_epoch'],'threshold':best['threshold'],'config_sha256':r['config_sha256'],'class_weight':w},run/'best_checkpoint.pt');r.update(best);r.update(fp);r.update({'status':'OK','wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(dev)/1024**2) if torch.cuda.is_available() else 0.,'parameter_count':sum(q.numel() for q in model.parameters()),'train_normal_count':normal,'train_anomaly_count':anomaly,'class_weight_normal':w[0],'class_weight_anomaly':w[1],'dropout':0.0,'reference_commit':got,'reference_file_sha256':REFSHA,'code_sha256':json.dumps(hashes(),sort_keys=True),'auroc_score':'softmax(logits)[:,1]'})
 except Exception:r['error']=traceback.format_exc()
 (run/'metrics.json').write_text(json.dumps(r,indent=2));csvout(P/'runs.csv',r);print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
