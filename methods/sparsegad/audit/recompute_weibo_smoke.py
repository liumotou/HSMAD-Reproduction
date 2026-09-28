"""Read-only independent recomputation for retained SparseGAD Weibo smoke."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from methods.project_paths import project_root
import dgl,torch
from methods.sparsegad.src.model import SparseGADModel
from methods.sparsegad.src.run_smoke import select
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
ROOT=project_root();SEED=ROOT/'results/experiments/sparsegad/weibo/sparsegad_h64_candidate/smoke/seed_0';OUT=SEED/'audit_recompute'
def main():
 OUT.mkdir(exist_ok=True);orig=json.loads((SEED/'metrics.json').read_text());state=torch.load(SEED/'checkpoint_last_epoch.pt',map_location='cpu');cfg=state['config'];raw=dgl.load_graphs(str(ROOT/cfg['dataset_file']))[0][0];g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));x=raw.ndata['feature'].float();y=raw.ndata['label'].long().reshape(-1);m={k:raw.ndata[k].bool() for k in ('train_mask','val_mask','test_mask')};model=SparseGADModel(x.shape[1],64,2,2,.2,.1);model.load_state_dict(state['model_state_dict']);model.eval()
 with torch.no_grad():p=torch.softmax(model(g,x),1)[:,1]
 t,v=select(y,p,m['val_mask']);truth=y[m['test_mask']].numpy();score=p[m['test_mask']].numpy();pred=(score>=t).astype(int);now={'f1_macro':float(f1_score(truth,pred,average='macro')),'auroc':float(roc_auc_score(truth,score)),'auprc':float(average_precision_score(truth,score)),'threshold':t,'predicted_anomaly_count':int(pred.sum()),'actual_anomaly_count':int(truth.sum())};comp={k:{'original':orig.get(k),'recomputed':now[k],'match':abs(orig[k]-now[k])<=1e-6 if isinstance(now[k],float) else orig[k]==now[k]} for k in now};status='recompute_match' if all(v['match'] for v in comp.values()) else 'recompute_mismatch';(OUT/'recompute_metrics.json').write_text(json.dumps(now,indent=2));(OUT/'compare_to_original.json').write_text(json.dumps({'status':status,'comparison':comp},indent=2));print(json.dumps({'status':status,'metrics':now}))
if __name__=='__main__':main()
