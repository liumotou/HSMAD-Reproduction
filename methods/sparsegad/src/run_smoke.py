"""One non-formal, bounded SparseGAD frozen-data smoke run."""
from __future__ import annotations
import argparse, hashlib, json, random, time
from datetime import datetime, timezone
from pathlib import Path
import dgl, numpy as np, torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from methods.sparsegad.src.model import SparseGADModel

ROOT = Path('/root/autodl-tmp/HSMAD')
THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 100, 5))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(tensor): return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(path, data): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); dgl.seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

def validate_smoke_config(config):
    """Keep smoke bounded while permitting only explicitly frozen datasets."""
    if config.get('dataset') not in {'weibo', 'tfinance'}:
        raise RuntimeError('unsupported frozen smoke dataset')
    if config.get('seed') != 0 or config.get('max_epoch') != 5 or config.get('run_type') != 'smoke':
        raise RuntimeError('fixed smoke contract violated')
def select(labels, probabilities, mask):
    truth, score = labels[mask].detach().cpu().numpy(), probabilities[mask].detach().cpu().numpy()
    scored = [(f1_score(truth, (score >= threshold).astype(np.int64), average='macro'), threshold) for threshold in THRESHOLDS]
    return max(scored)[1], max(scored)[0]
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config', required=True); args=parser.parse_args()
    config_path=Path(args.config).resolve(); config=json.loads(config_path.read_text()); output=ROOT/config['result_dir']
    if output.exists(): raise FileExistsError(output)
    validate_smoke_config(config)
    seed(0); raw_path=ROOT/config['dataset_file']; raw=dgl.load_graphs(str(raw_path))[0][0]
    graph=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    features,labels=raw.ndata['feature'].float(),raw.ndata['label'].long().reshape(-1)
    masks={key:raw.ndata[key].bool() for key in ('train_mask','val_mask','test_mask')}
    if graph.num_nodes()!=config['expected']['nodes'] or graph.num_edges()!=config['expected']['training_edges']: raise RuntimeError('frozen graph mismatch')
    hashes={'dataset_file_sha256':sha(raw_path),'feature_sha256':tensor_sha(features),'label_sha256':tensor_sha(labels), **{f'{key}_sha256':tensor_sha(value) for key,value in masks.items()}, 'model_py_sha256':sha(ROOT/'methods/sparsegad/src/model.py'),'runner_py_sha256':sha(Path(__file__)),'config_sha256':sha(config_path)}
    output.mkdir(parents=True); write(output/'config_snapshot.json',config)
    write(output/'preflight.json',{'passed':True,'raw_graph':{'nodes':raw.num_nodes(),'edges':raw.num_edges()},'training_graph':{'nodes':graph.num_nodes(),'edges':graph.num_edges()},'feature_shape':list(features.shape),'mask_counts':{key:int(value.sum()) for key,value in masks.items()},'edge_access':config['edge_access'],'hashes':hashes})
    device=torch.device('cuda'); graph,features,labels=graph.to(device),features.to(device),labels.to(device); masks={key:value.to(device) for key,value in masks.items()}
    model=SparseGADModel(features.shape[1],config['hidden_dim'],2,config['num_layers'],config['dropout'],config['dropout_adj']).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    torch.cuda.reset_peak_memory_stats(device); start=time.monotonic(); history=[]
    for epoch in range(1,6):
        model.train(); logits=model(graph,features); loss=torch.nn.functional.cross_entropy(logits[masks['train_mask']],labels[masks['train_mask']]); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval();
        with torch.no_grad(): probability=torch.softmax(model(graph,features),1)[:,1]
        threshold,val_f1=select(labels,probability,masks['val_mask']); truth=labels[masks['val_mask']].cpu().numpy(); score=probability[masks['val_mask']].cpu().numpy()
        record={'epoch':epoch,'train_loss':float(loss),'validation_f1_macro':val_f1,'validation_auroc':float(roc_auc_score(truth,score)),'validation_auprc':float(average_precision_score(truth,score)),'validation_threshold':threshold,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(device)/1024**2)}; history.append(record); print(json.dumps(record),flush=True)
    torch.save({'epoch':5,'model_state_dict':model.state_dict(),'config':config},output/'checkpoint_last_epoch.pt')
    with torch.no_grad(): probability=torch.softmax(model(graph,features),1)[:,1]
    threshold,val_f1=select(labels,probability,masks['val_mask']); truth=labels[masks['test_mask']].cpu().numpy(); score=probability[masks['test_mask']].cpu().numpy(); prediction=(score>=threshold).astype(np.int64)
    metrics={'method':config['method'],'protocol_version':config['protocol_version'],'positioning':config['positioning'],'dataset':config['dataset'],'seed':0,'run_type':'smoke','status':'smoke','actual_epochs':5,'f1_macro':float(f1_score(truth,prediction,average='macro')),'auroc':float(roc_auc_score(truth,score)),'auprc':float(average_precision_score(truth,score)),'threshold':threshold,'validation_f1_macro':val_f1,'predicted_anomaly_count':int(prediction.sum()),'actual_anomaly_count':int(truth.sum()),'wall_time_sec':time.monotonic()-start,'peak_gpu_mb':float(torch.cuda.max_memory_allocated(device)/1024**2),'edge_access':config['edge_access'],'hashes':hashes,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write(output/'validation_history.json',history); write(output/'metrics.json',metrics); write(output/'runs.json',metrics); write(output/'artifact_sha256s.json',{path.name:sha(path) for path in output.iterdir() if path.is_file()}); print('SMOKE_COMPLETE '+json.dumps(metrics),flush=True)
if __name__=='__main__': main()
