"""One-seed, feature-only dropout-zero probe; it never mutates v1/v2."""
import argparse, csv, hashlib, json, shutil, subprocess, sys, time, traceback
from pathlib import Path
import dgl, numpy as np, torch
import torch.nn.functional as functional
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.train import EXPECTED_WEIBO_MASKS,best_validation_threshold,load_feature_data
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
PROBE=ROOT/'results/experiments/mlp/weibo/protocol_v3_dropout_zero_probe'; V1=ROOT/'results/experiments/mlp/weibo/formal/seed_0'
REF_COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6';REF_SHA='6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d'

def canonical_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def code_hashes():return {f:file_sha256(str(ROOT/f)) for f in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
def environment(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def semantic_v1(c):return {k:c[k] for k in ['model','dataset','architecture','activation','hidden_dim','dropout','loss','class_weight','sampling','optimizer','learning_rate','weight_decay','max_epoch','patience','selection_metric','threshold_candidates','edge_access']}
def semantic_v3(c):return {k:c[k] for k in ['model','dataset','architecture','activation','hidden_dim','dropout','loss','class_weight','sampling','optimizer','learning_rate','weight_decay','max_epoch','patience','selection_metric','threshold_candidates','edge_access']}
def allowed_v1_v3_diff(v1,v3):return {k:{'v1':v1[k],'v3':v3[k]} for k in v1 if v1[k]!=v3[k]}
def write_csv(p,row):
 with p.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(row));w.writeheader();w.writerow(row)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--config',required=True);a=p.parse_args()
 if a.seed!=0:raise ValueError('Only seed=0 is authorized')
 config_path=Path(a.config).resolve();c=json.loads(config_path.read_text());v1c=json.loads((ROOT/'methods/mlp/configs/weibo_formal.json').read_text());diff=allowed_v1_v3_diff(semantic_v1(v1c),semantic_v3(c))
 if diff!={'dropout':{'v1':0.5,'v3':0.0}}:raise RuntimeError('v1/v3 diff is not dropout-only: '+json.dumps(diff,sort_keys=True))
 run=PROBE/'seed_0';run.mkdir(parents=True,exist_ok=False);envd=run/'environment';envd.mkdir();record={'method':'MLP','dataset':'weibo','seed':0,'run_type':'protocol_v3_dropout_zero_probe','status':'ERROR','protocol_version':'v3','edge_access':'none','config_sha256':canonical_sha(c)}
 try:
  setup_seed(0);(PROBE/'config_diff_preflight.json').write_text(json.dumps({'status':'PASS','only_semantic_training_difference':'dropout','v1_dropout':0.5,'v3_dropout':0.0,'v1_class_weight':None,'v3_class_weight':None,'diff':diff},indent=2))
  dataset=ROOT/'datasets/weibo';x,y,m=load_feature_data(dataset);fp={'dataset_file_sha256':file_sha256(str(dataset)),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()}}
  for k,e in EXPECTED_WEIBO_MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('mask mismatch '+k)
  v1=json.loads((V1/'metrics.json').read_text())
  for k in fp:
   if fp[k]!=v1[k]:raise RuntimeError('v1 input mismatch '+k)
  ref=ROOT/'audit/mlp_reference/GADBench/models/gnn.py'
  if file_sha256(str(ref))!=REF_SHA:raise RuntimeError('reference sha mismatch')
  commit=subprocess.run(['git','-C',str(ref.parents[1]),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
  if commit!=REF_COMMIT:raise RuntimeError('reference commit mismatch')
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','reference_commit':commit,'reference_file_sha256':REF_SHA,'input_sha256':fp,'code_sha256':code_hashes(),'environment':environment(d),'v1_v3_config_diff':diff,'class_weight':None,'edge_access':'none'}
  (run/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(config_path,run/'config_snapshot.json');(envd/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(envd/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(d),y.to(d);m={k:v.to(d) for k,v in m.items()};model=FeatureMLP(x.shape[1],64,0.0).to(d);opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);best=None;state=None;stalled=0;start=time.perf_counter()
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(d)
  for ep in range(1,c['max_epoch']+1):
   model.train();opt.zero_grad();loss=functional.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():prob=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vf,t=best_validation_threshold(y[m['val_mask']].cpu().numpy(),prob[vm])
   if best is None or vf>best['val_f1']:
    tm=m['test_mask'].cpu().numpy();tl=y[m['test_mask']].cpu().numpy();tp=prob[tm];pr=(tp>t).astype(np.int64);best={'val_f1':vf,'threshold':t,'best_epoch':ep,'f1_macro':float(f1_score(tl,pr,average='macro')),'auroc':float(roc_auc_score(tl,np.nan_to_num(tp))),'test_predicted_anomaly_count':int(pr.sum()),'test_predicted_anomaly_ratio':float(pr.mean()),'test_confusion_matrix_labels_0_1':confusion_matrix(tl,pr,labels=[0,1]).tolist()};state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stalled=0
   else:stalled+=1
   if stalled>=c['patience']:break
  torch.save({'model_state_dict':state,'seed':0,'best_epoch':best['best_epoch'],'threshold':best['threshold'],'config_sha256':record['config_sha256'],'class_weight':None},run/'best_checkpoint.pt');record.update(best);record.update(fp);record.update({'status':'OK','wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2) if torch.cuda.is_available() else 0.0,'parameter_count':sum(z.numel() for z in model.parameters()),'class_weight':None,'dropout':0.0,'reference_commit':commit,'reference_file_sha256':REF_SHA,'code_sha256':json.dumps(code_hashes(),sort_keys=True),'auroc_score':'softmax(logits)[:,1]'})
 except Exception:record['error']=traceback.format_exc()
 (run/'metrics.json').write_text(json.dumps(record,indent=2));write_csv(PROBE/'runs.csv',record);print(json.dumps(record,sort_keys=True))
 if record['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
