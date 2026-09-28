from __future__ import annotations
import csv, json, statistics
from pathlib import Path
ROOT=Path('/root/autodl-tmp/HSMAD')
BASE=ROOT/'results/experiments/bwgnn/weibo/bwgnn_gadbench_h64_candidate/formal'
rows=[]
for seed in range(10):
    metrics=json.loads((BASE/f'seed_{seed}'/'metrics.json').read_text())
    audit=json.loads((BASE/f'seed_{seed}'/'audit_recompute'/'compare_to_original.json').read_text())
    metrics['audit_status']=audit['status']; rows.append(metrics)
def values(key): return [row[key] for row in rows if row['status']=='OK' and row['audit_status']=='recompute_match']
summary={'method':'BWGNN-GADBench-h64','dataset':'weibo','positioning':'candidate_protocol_not_author_exact','successful_audited_seeds':len(values('f1_macro')),'f1_macro_mean':statistics.mean(values('f1_macro')),'f1_macro_std_sample':statistics.stdev(values('f1_macro')),'auroc_mean':statistics.mean(values('auroc')),'auroc_std_sample':statistics.stdev(values('auroc')),'failed_seeds':[row['seed'] for row in rows if row['status']!='OK' or row['audit_status']!='recompute_match']}
(BASE/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
(BASE/'formal_audit.md').write_text('# BWGNN Weibo formal candidate audit\n\n'+json.dumps(summary,indent=2)+'\n')
with (BASE/'audited_runs.csv').open('w',newline='') as f:
    fields=['seed','status','audit_status','f1_macro','auroc','auprc','best_epoch','threshold','wall_time_sec','peak_gpu_mb'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
print(json.dumps(summary,sort_keys=True))
