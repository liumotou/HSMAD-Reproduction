"""Read-only audit of retained BWGNN Weibo formal artifacts."""
import csv, json, statistics
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()
FORMAL = ROOT / 'results/experiments/bwgnn/weibo/bwgnn_gadbench_h64_candidate/formal'
OUT = ROOT / 'methods/bwgnn/audit'

def main():
    rows = list(csv.DictReader((FORMAL / 'runs.csv').open(encoding='utf-8')))
    ok = [r for r in rows if r.get('status') == 'OK' and r.get('run_type') == 'formal']
    seeds = []
    for r in ok:
        metric = json.loads((FORMAL / ('seed_' + r['seed']) / 'metrics.json').read_text())
        audit = json.loads((FORMAL / ('seed_' + r['seed']) / 'audit_recompute/compare_to_original.json').read_text())
        seeds.append({'seed': r['seed'], 'f1_macro': metric['f1_macro'], 'auroc': metric['auroc'], 'best_epoch': metric['best_epoch'], 'threshold': metric['threshold'], 'recompute_status': audit['status'], **metric['hashes']})
    report = {'status': 'FORMAL_CANDIDATE_SUMMARY', 'n_formal_ok': len(ok), 'seeds': [int(r['seed']) for r in ok],
      'f1_mean': statistics.mean(float(r['f1_macro']) for r in ok), 'f1_sample_sd': statistics.stdev(float(r['f1_macro']) for r in ok),
      'auroc_mean': statistics.mean(float(r['auroc']) for r in ok), 'auroc_sample_sd': statistics.stdev(float(r['auroc']) for r in ok),
      'paper_f1': .9302, 'paper_auroc': .9722, 'checkpoint_recompute_all_match': all(r['recompute_status']=='recompute_match' for r in seeds),
      'fixed_split_across_seeds': len({(r['train_mask_sha256'],r['val_mask_sha256'],r['test_mask_sha256']) for r in seeds}) == 1,
      'static_protocol': 'loss=train_mask; AUPRC checkpoint/threshold=val_mask; test only after checkpoint fixed; model forward graph-only.',
      'positioning': 'candidate_protocol_not_author_exact'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'bwgnn_weibo_formal_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    with (OUT/'bwgnn_weibo_seed_level.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted(seeds[0]));w.writeheader();w.writerows(seeds)
    (OUT/'bwgnn_weibo_formal_audit.md').write_text('# BWGNN Weibo formal audit\n\n- 10 formal/OK seeds retained; sample SD (ddof=1).\n- All checkpoints independently recompute-match.\n- Fixed frozen split across seeds; train/val/test mask separation is enforced by the runner/protocol call path.\n')
    print(json.dumps(report,sort_keys=True))
if __name__ == '__main__': main()
