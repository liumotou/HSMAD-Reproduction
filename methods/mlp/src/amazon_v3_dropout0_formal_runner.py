"""Isolated formal MLP Amazon runner; it uses the frozen v3 diagnostic config verbatim."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import dgl,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'methods/mlp/src'))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.utils import file_sha256,setup_seed,tensor_sha256
OUT=ROOT/'results/experiments/mlp/amazon/protocol_v3_dropout0/formal'
MASKS={'train_mask':'fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a','val_mask':'2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb','test_mask':'bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1'}
COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6';REFSHA='6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d'
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def codehash():return {f:file_sha256(str(ROOT/f)) for f in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
def env(d):return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,'cuda_available':torch.cuda.is_available(),'device':str(d),'gpu_name':torch.cuda.get_device_name(d) if torch.cuda.is_available() else None}
def threshold(y,p):
 b=(-1.,.05)
 for t in np.linspace(.05,.95,19):
  v=f1_score(y,(p>t).astype(np.int64),average='macro')
  if v>b[0]:b=(v,float(t))
 return b
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',required=True,type=int,choices=range(10));a.add_argument('--config',required=True);z=a.parse_args();cp=Path(z.config).resolve();c=json.loads(cp.read_text());run=OUT/f'seed_{z.seed}';run.mkdir(parents=True,exist_ok=False);ed=run/'environment';ed.mkdir();r={'method':'MLP','dataset':'amazon','seed':z.seed,'run_type':'formal','protocol_version':'v3_dropout0','status':'ERROR','edge_access':'none','config_sha256':csha(c)}
 try:
  setup_seed(z.seed);graphs,_=dgl.load_graphs(str(ROOT/'datasets/amazon'));n=graphs[0].ndata;x=n['feature'].float().contiguous();y=n['label'].long().reshape(-1).contiguous();m={k:n[k].bool().reshape(-1).contiguous() for k in MASKS};fp={'dataset_file_sha256':file_sha256(str(ROOT/'datasets/amazon')),'feature_sha256':tensor_sha256(x),'label_sha256':tensor_sha256(y),**{k+'_sha256':tensor_sha256(v) for k,v in m.items()},'nodes':int(y.numel()),'feature_shape':list(x.shape)}
  for k,e in MASKS.items():
   if fp[k+'_sha256']!=e:raise RuntimeError('frozen HSMAD mask SHA mismatch '+k)
  prefix=int(c['amazon_excluded_prefix_nodes']);covered=m['train_mask']|m['val_mask']|m['test_mask'];pc={k:int(v[:prefix].sum()) for k,v in m.items()}
  if any(pc.values()) or int(covered[:prefix].sum())!=0:raise RuntimeError('official Amazon excluded-prefix rule violated')
  if x.shape!=(y.numel(),25) or c['input_dim']!=25:raise RuntimeError('unexpected Amazon feature shape')
  ref=ROOT/'audit/mlp_reference/GADBench/models/gnn.py'
  if file_sha256(str(ref))!=REFSHA:raise RuntimeError('reference file SHA mismatch')
  got=subprocess.run(['git','-C',str(ref.parents[1]),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
  if got!=COMMIT:raise RuntimeError('reference commit mismatch')
  d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');pre={'status':'PASS','input_sha256':fp,'frozen_hsmad_masks_match':True,'amazon_excluded_prefix_nodes':prefix,'prefix_mask_counts':pc,'prefix_covered_count':int(covered[:prefix].sum()),'config_sha256':r['config_sha256'],'reference_commit':got,'reference_file_sha256':REFSHA,'code_sha256':codehash(),'environment':env(d),'edge_access':'none','edge_access_proof':'runner only reads graph.ndata; no edge accessor, preprocessing, aggregation, or message passing'}
  (run/'preflight.json').write_text(json.dumps(pre,indent=2));shutil.copy2(cp,run/'config_snapshot.json');(ed/'framework_versions.json').write_text(json.dumps(pre['environment'],indent=2));(ed/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
  x,y=x.to(d),y.to(d);m={k:v.to(d) for k,v in m.items()};model=FeatureMLP(25,64,0.).to(d);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5);best=None;state=None;stall=0;hist=[]
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(d)
  start=time.perf_counter()
  for ep in range(1,1001):
   model.train();opt.zero_grad();loss=F.cross_entropy(model(x)[m['train_mask']],y[m['train_mask']]);loss.backward();opt.step();model.eval()
   with torch.no_grad():p=torch.softmax(model(x),dim=1)[:,1].cpu().numpy()
   vm=m['val_mask'].cpu().numpy();vf,t=threshold(y[m['val_mask']].cpu().numpy(),p[vm]);hist.append({'epoch':ep,'loss':float(loss.detach().cpu()),'val_f1_macro':vf,'threshold':t})
   if best is None or vf>best['val_f1']:
    tm=m['test_mask'].cpu().numpy();tl=y[m['test_mask']].cpu().numpy();tp=p[tm];pd=(tp>t).astype(np.int64);best={'val_f1':vf,'threshold':t,'best_epoch':ep,'f1_macro':float(f1_score(tl,pd,average='macro')),'auroc':float(roc_auc_score(tl,np.nan_to_num(tp))),'test_predicted_anomaly_count':int(pd.sum()),'test_predicted_anomaly_ratio':float(pd.mean()),'test_confusion_matrix_labels_0_1':confusion_matrix(tl,pd,labels=[0,1]).tolist()};state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stall=0
   else:stall+=1
   if stall>=100:break
  (run/'validation_history.json').write_text(json.dumps(hist,indent=2));torch.save({'model_state_dict':state,'seed':z.seed,'best_epoch':best['best_epoch'],'threshold':best['threshold'],'config_sha256':r['config_sha256']},run/'best_checkpoint.pt');r.update(best);r.update(fp);r.update({'status':'OK','wall_time_sec':time.perf_counter()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(d)/1024**2) if torch.cuda.is_available() else 0.,'parameter_count':sum(q.numel() for q in model.parameters()),'class_weight':None,'dropout':0.0,'reference_commit':got,'reference_file_sha256':REFSHA,'code_sha256':json.dumps(codehash(),sort_keys=True),'auroc_score':'softmax(logits)[:,1]','paper_f1_delta':best['f1_macro']-.9223,'paper_auroc_delta':best['auroc']-.9801})
 except Exception:r['error']=traceback.format_exc()
 (run/'result.csv').write_text(','.join(r.keys())+'\n'+','.join(json.dumps(v) if isinstance(v,(dict,list)) else str(v) for v in r.values())+'\n');(run/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,sort_keys=True))
 if r['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
