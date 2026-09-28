"""One-seed, feature-only class-weight probe. No v1 artifact is modified."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'methods/mlp/src'))
from model import FeatureMLP
from train import EXPECTED_WEIBO_MASKS, best_validation_threshold, load_feature_data
from utils import file_sha256, setup_seed, tensor_sha256

PROBE = ROOT / 'results/experiments/mlp/weibo/protocol_v2_class_weight_probe'
V1_DIR = ROOT / 'results/experiments/mlp/weibo/formal/seed_0'
REFERENCE_COMMIT='f9aa021ce9b6c6580427fb633b596843be76ddc6'
REFERENCE_SHA='6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d'

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def canonical_sha(obj): return sha_bytes(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())
def code_hashes():
    files=['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']
    return {f:file_sha256(str(ROOT/f)) for f in files}
def environment(device):
    return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'dgl':dgl.__version__,
            'cuda_available':torch.cuda.is_available(),'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if torch.cuda.is_available() else None}
def write_csv(path,row):
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(row));w.writeheader();w.writerow(row)
def semantic_v1_projection(v1):
    return {'model':v1['model'],'dataset':v1['dataset'],'architecture':v1['architecture'],'activation':v1['activation'],
      'hidden_dim':v1['hidden_dim'],'dropout':v1['dropout'],'loss':v1['loss'],'sampling':v1['sampling'],
      'optimizer':v1['optimizer'],'learning_rate':v1['learning_rate'],'weight_decay':v1['weight_decay'],
      'max_epoch':v1['max_epoch'],'patience':v1['patience'],'selection_metric':v1['selection_metric'],
      'threshold_candidates':v1['threshold_candidates'],'edge_access':v1['edge_access']}
def semantic_v2_projection(v2):
    return {'model':v2['model'],'dataset':v2['dataset'],'architecture':v2['architecture'],'activation':v2['activation'],
      'hidden_dim':v2['hidden_dim'],'dropout':v2['dropout'],'loss':v2['loss'],'sampling':v2['sampling'],
      'optimizer':v2['optimizer'],'learning_rate':v2['learning_rate'],'weight_decay':v2['weight_decay'],
      'max_epoch':v2['max_epoch'],'patience':v2['patience'],'selection_metric':v2['selection_metric'],
      'threshold_candidates':v2['threshold_candidates'],'edge_access':v2['edge_access']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--config',required=True);a=p.parse_args()
    if a.seed != 0: raise ValueError('Probe authorization permits seed=0 only')
    config_path=Path(a.config).resolve(); config=json.loads(config_path.read_text()); config_sha=canonical_sha(config)
    run=PROBE/'seed_0'; run.mkdir(parents=True,exist_ok=False); envdir=run/'environment';envdir.mkdir()
    record={'method':'MLP','dataset':'weibo','seed':0,'run_type':'protocol_v2_class_weight_probe','status':'ERROR','protocol_version':'v2','edge_access':'none','config_sha256':config_sha}
    try:
        setup_seed(0)  # intentionally before all data/model/optimizer creation
        v1config=json.loads((ROOT/'methods/mlp/configs/weibo_formal.json').read_text())
        projection_diff={key:{'v1':semantic_v1_projection(v1config)[key],'v2':semantic_v2_projection(config)[key]} for key in semantic_v1_projection(v1config) if semantic_v1_projection(v1config)[key]!=semantic_v2_projection(config)[key]}
        if projection_diff:
            raise RuntimeError('non-class-weight config difference: '+json.dumps(projection_diff,sort_keys=True))
        config_diff={'status':'PASS','only_semantic_training_difference':'class_weight','unchanged_v1_v2':semantic_v1_projection(v1config),
          'v1_class_weight':None,'v2_class_weight':config['class_weight'],'difference_count':1}
        (PROBE/'config_diff_preflight.json').write_text(json.dumps(config_diff,indent=2))
        dataset=ROOT/'datasets/weibo'; features,labels,masks=load_feature_data(dataset)
        fps={'dataset_file_sha256':file_sha256(str(dataset)),'feature_sha256':tensor_sha256(features),'label_sha256':tensor_sha256(labels),**{k+'_sha256':tensor_sha256(v) for k,v in masks.items()}}
        for key,expected in EXPECTED_WEIBO_MASKS.items():
            if fps[key+'_sha256']!=expected: raise RuntimeError('frozen '+key+' mismatch')
        v1=json.loads((V1_DIR/'metrics.json').read_text())
        for key in ('dataset_file_sha256','feature_sha256','label_sha256','train_mask_sha256','val_mask_sha256','test_mask_sha256'):
            if fps[key]!=v1[key]: raise RuntimeError('v1 input mismatch '+key)
        ref=ROOT/'audit/mlp_reference/GADBench/models/gnn.py'
        if file_sha256(str(ref)) != REFERENCE_SHA: raise RuntimeError('reference SHA mismatch')
        commit=subprocess.run(['git','-C',str(ref.parents[1]),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
        if commit != REFERENCE_COMMIT: raise RuntimeError('reference commit mismatch')
        train_labels=labels[masks['train_mask']]
        normal=int((train_labels==0).sum()); anomaly=int((train_labels==1).sum())
        if anomaly==0: raise RuntimeError('zero train anomalies')
        weights=[1.0,normal/anomaly]
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        preflight={'status':'PASS','v1_config_diff_only':'class_weight','v1_metrics_sha256':file_sha256(str(V1_DIR/'metrics.json')),
                   'reference_commit':commit,'reference_file_sha256':REFERENCE_SHA,'input_sha256':fps,'code_sha256':code_hashes(),
                   'environment':environment(device),'train_normal_count':normal,'train_anomaly_count':anomaly,'class_weight':weights,
                   'class_weight_source':'GADBench reference commit '+REFERENCE_COMMIT,
                   'claim_boundary':'This does not establish HSMAD authors original Table 1 MLP configuration.','edge_access':'none'}
        (run/'preflight.json').write_text(json.dumps(preflight,indent=2));shutil.copy2(config_path,run/'config_snapshot.json')
        (envdir/'framework_versions.json').write_text(json.dumps(preflight['environment'],indent=2))
        (envdir/'packages_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout)
        features,labels=features.to(device),labels.to(device);masks={k:v.to(device) for k,v in masks.items()}
        class_weight=torch.tensor(weights,dtype=torch.float32,device=device)
        model=FeatureMLP(features.shape[1],config['hidden_dim'],config['dropout']).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        best=None;best_state=None;stalled=0;started=time.perf_counter()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1,config['max_epoch']+1):
            model.train();optimizer.zero_grad();logits=model(features)
            loss=functional.cross_entropy(logits[masks['train_mask']],labels[masks['train_mask']],weight=class_weight)
            loss.backward();optimizer.step();model.eval()
            with torch.no_grad(): probs=torch.softmax(model(features),dim=1)[:,1].cpu().numpy()
            vm=masks['val_mask'].cpu().numpy();vl=labels[masks['val_mask']].cpu().numpy();vf,thr=best_validation_threshold(vl,probs[vm])
            if best is None or vf>best['val_f1']:
                tm=masks['test_mask'].cpu().numpy();tl=labels[masks['test_mask']].cpu().numpy();tp=probs[tm];pred=(tp>thr).astype(np.int64)
                best={'val_f1':vf,'threshold':thr,'best_epoch':epoch,'f1_macro':float(f1_score(tl,pred,average='macro')),'auroc':float(roc_auc_score(tl,np.nan_to_num(tp))),
                      'test_predicted_anomaly_count':int(pred.sum()),'test_predicted_anomaly_ratio':float(pred.mean()),'test_confusion_matrix_labels_0_1':confusion_matrix(tl,pred,labels=[0,1]).tolist()}
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stalled=0
            else: stalled+=1
            if stalled>=config['patience']:break
        torch.save({'model_state_dict':best_state,'seed':0,'best_epoch':best['best_epoch'],'threshold':best['threshold'],'config_sha256':config_sha,'class_weight':weights},run/'best_checkpoint.pt')
        record.update(best);record.update(fps);record.update({'status':'OK','wall_time_sec':time.perf_counter()-started,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(device)/1024**2) if torch.cuda.is_available() else 0.0,
          'parameter_count':sum(x.numel() for x in model.parameters()),'train_normal_count':normal,'train_anomaly_count':anomaly,'class_weight_normal':weights[0],'class_weight_anomaly':weights[1],
          'class_weight_source':'GADBench reference commit '+REFERENCE_COMMIT,'class_weight_claim_boundary':'Not evidence of HSMAD Table 1 original MLP configuration.',
          'reference_commit':commit,'reference_file_sha256':REFERENCE_SHA,'code_sha256':json.dumps(code_hashes(),sort_keys=True),'auroc_score':'softmax(logits)[:,1]'})
    except Exception: record['error']=traceback.format_exc()
    (run/'metrics.json').write_text(json.dumps(record,indent=2));write_csv(PROBE/'runs.csv',record);print(json.dumps(record,sort_keys=True))
    if record['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
