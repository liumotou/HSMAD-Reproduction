"""Read-only metric recomputation for the retained BWGNN Weibo smoke checkpoint."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import dgl, torch
from methods.bwgnn.src.model import GADBenchBWGNN
from methods.bwgnn.src.protocol import select_validation_threshold, test_metrics

ROOT = Path('/root/autodl-tmp/HSMAD')
DEFAULT_SEED = ROOT / 'results/experiments/bwgnn/weibo/bwgnn_gadbench_h64_candidate/smoke_retry_01/seed_0'

def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=str(DEFAULT_SEED))
    parser.add_argument('--audit-name', default='audit_recompute_tolerance_1e6')
    args=parser.parse_args()
    seed=Path(args.run_dir)
    out=seed/args.audit_name
    out.mkdir(exist_ok=True)
    original=json.loads((seed/'metrics.json').read_text())
    checkpoint=torch.load(seed/'checkpoint_auprc_best.pt' if (seed/'checkpoint_auprc_best.pt').exists() else seed/'checkpoint_last_epoch.pt', map_location='cpu')
    config=checkpoint['config']
    raw=dgl.load_graphs(str(ROOT/config['dataset_file']))[0][0]
    graph=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    graph.ndata['feature']=raw.ndata['feature'].float()
    label=raw.ndata['label'].long().reshape(-1)
    masks={key:raw.ndata[key].bool() for key in ('train_mask','val_mask','test_mask')}
    model=GADBenchBWGNN(raw.ndata['feature'].shape[1], 64, 2, 2, 2, 0.0)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    with torch.no_grad(): probability=torch.softmax(model(graph), dim=1)[:,1]
    threshold, val_f1=select_validation_threshold(label, probability, masks['val_mask'])
    result=test_metrics(label, probability, masks['test_mask'], threshold)
    observed={'threshold':threshold,'validation_f1_macro':val_f1, **result,
              'checkpoint_sha256':digest(seed/('checkpoint_auprc_best.pt' if (seed/'checkpoint_auprc_best.pt').exists() else 'checkpoint_last_epoch.pt')),
              'feature_sha256':hashlib.sha256(raw.ndata['feature'].contiguous().numpy().tobytes()).hexdigest(),
              'label_sha256':hashlib.sha256(label.contiguous().numpy().tobytes()).hexdigest(),
              'mask_counts':{key:int(value.sum()) for key,value in masks.items()}}
    def same(key):
        left, right = original.get(key), observed.get(key)
        return abs(left-right) <= 1e-6 if isinstance(left, float) and isinstance(right, float) else left == right
    compare={key:{'original':original.get(key),'recomputed':observed.get(key),'match':same(key)} for key in ('f1_macro','auroc','auprc','threshold','predicted_anomaly_count','actual_anomaly_count','confusion_matrix')}
    status='recompute_match' if all(item['match'] for item in compare.values()) else 'recompute_mismatch'
    (out/'recompute_metrics.json').write_text(json.dumps(observed,indent=2,sort_keys=True))
    (out/'compare_to_original.json').write_text(json.dumps({'status':status,'comparison':compare},indent=2,sort_keys=True))
    (out/'audit.md').write_text('# BWGNN Weibo independent recomputation\n\nStatus: '+status+'\n')
    print(json.dumps({'status':status,'metrics':observed},sort_keys=True))
if __name__=='__main__': main()
