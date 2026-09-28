"""Amazon seed-0 MLP diagnostic: feature-only, frozen masks, no graph edges."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
OUT=ROOT/'results/experiments/mlp/amazon/protocol_v1_seed0_diagnostic'
MASKS={'train_mask':'fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a','val_mask':'2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb','test_mask':'bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1'}
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def best(y,p):
 out=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  s=f1_score(y,(p>t).astype(np.int64),average='macro')
  if s>out[0]:out=(s,float(t))
 return out
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def hashes():return {f:file_sha256(str(ROOT/f)) for f in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
def writecsv(p,row):
 with p.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(row));w.writeheader();w.writerow(row)
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,default=0);a.add_argument('--config',required=True);z=a.parse_args()
 if z.seed!=0:raise ValueError('diagnostic authorization permits seed=0 only')
 cp=Path(z.config).resolve();c=json.loads(cp.read_text());run=OUT/'seed_0';run.mkdir(parents=True,exist_ok=False);ed=run/'environment';ed.mkdir();r={'method':'MLP','dataset':'amazon','seed':0,'run_type':'diagnostic','status':'ERROR','edge_access':'none','config_sha256':csha(c)}
 try:
  setup_seed(0)
  gs,_=dgl.load_graphs(str(ROOT/'datasets/amazon'));n=gs[0].ndata;x=n['feature'].float().contiguous();y=n['label'].long().reshape(-1).contiguous();m={k:n[k].bool().reshape(-1).contiguous() for k in MASKS}
  fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/amazon')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()},'nodes':int(y.numel()),'feature_shape':list(x.shape)}
  for k,e in MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('HSMAD frozen '+k+' SHA mismatch')
  prefix=int(c['amazon_excluded_prefix_nodes']); covered=m['train_mask']|m['val_mask']|m['test_mask']; prefix_counts={k:int(v[:prefix].sum()) for k,v in m.items()}
  if any(prefix_counts.values()):raise RuntimeError('Amazon excluded prefix leaked into a split')
  if int(covered[:prefix].sum())!=0:raise RuntimeError('excluded prefix appears in evaluation')
  if x.shape[1]!=c['input_dim'] or x.shape[1]!=25:raise RuntimeError('unexpected Amazon feature dimension')
  v1=json.loads((ROOT/'methods/mlp/configs/weibo_formal.json').read_text());shared=['model','architecture','activation','hidden_dim','dropout','loss','class_weight','sampling','optimizer','learning_rate','weight_decay','max_epoch','patience','selection_metric','threshold_candidates','edge_access'];diff={k:{'weibo_v1':v1[k],'amazon_diag':c[k]} for k in shared if v1[k]!=c[k]}
  if diff:raise RuntimeError('non-dataset v1 config mismatch '+json.dumps(diff,sort_keys=True))
  dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','input_sha256':fp,'hsmad_mask_sha256_match':True,'amazon_excluded_prefix_nodes':prefix,'prefix_mask_counts':prefix_counts,'prefix_covered_count':int(covered[:prefix].sum()),'only_data_input_config_difference_from_weibo_v1':True,'semantic_config_diff':{'dataset':'weibo -> amazon','input_dim':'400 -> 25'},'code_sha256':hashes(),'environment':env(dev),'edge_access':'none'}
  (OUT/'config_diff_preflight.json').write_text(json.dumps(pre,indent=2));(run/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,run/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(dev),y.to(dev);m={k:v.to(dev) for k,v in m.items()};model=FeatureMLP(25,64,.5).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);bestrec=None;state=None;stall=0;history=[];start=time.perf_counter()
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(dev)
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():prob=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vf,t=best(y[m['val_mask']].cpu().numpy(),prob[vm]);history.append({'epoch':ep,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'threshold':t})
   if bestrec is None or vf>bestrec['val_f1']:
    tm=m['test_mask'].cpu().numpy();tl=y[m['test_mask']].cpu().numpy();tp=prob[tm];pd=(tp>t).astype(np.int64);bestrec={'val_f1':vf,'threshold':t,'best_epoch':ep,'f1_macro':float(f1_score(tl,pd,average='macro')),'auroc':float(roc_auc_score(tl,np.nan_to_num(tp))),'test_predicted_anomaly_count':int(pd.sum()),'test_predicted_anomaly_ratio':float(pd.mean()),'test_confusion_matrix_labels_0_1':confusion_matrix(tl,pd,labels=[0,1]).tolist()};state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stall=0
   else:stall+=1
   if stall>=100:break
  (run/'validation_history.json').write_text(json.dumps(history,indent=2));torch.save({'model_state_dict':state,'seed':0,'best_epoch':bestrec['best_epoch'],'threshold':bestrec['threshold'],'config_sha256':r['config_sha256']},run/'best_checkpoint.pt');r.update(bestrec);r.update(fp);r.update({'status':'OK','wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(dev)/1024**2) if torch.cuda.is_available() else 0.,'parameter_count':sum(p.numel() for p in model.parameters()),'prefix_mask_counts':prefix_counts,'paper_f1_delta':bestrec['f1_macro']-.9223,'paper_auroc_delta':bestrec['auroc']-.9801,'auroc_score':'softmax(logits)[:,1]','class_weight':None,'dropout':.5})
 except Exception:r['error']=traceback.format_exc()
 (run/'metrics.json').write_text(json.dumps(r,indent=2));writecsv(OUT/'runs.csv',r);print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
