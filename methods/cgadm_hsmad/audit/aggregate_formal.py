"""Aggregate only CGADM formal/OK seed results with sample standard deviation."""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--formal-dir',type=Path,required=True)
p.add_argument('--dataset',default='weibo')
a=p.parse_args()
rows=[]
for seed in range(10):
    path=a.formal_dir/f'seed_{seed}'/'metrics.json'
    if not path.exists(): continue
    row=json.loads(path.read_text())
    if row.get('run_type')=='formal' and row.get('status')=='OK':
        row['metrics_sha256']=sha(path); rows.append(row)
fields=sorted({k for r in rows for k in r})
with (a.formal_dir/'runs.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
summary={'method':'CGADM','dataset':a.dataset,'protocol':'cgadm_hsmad_candidate','n':len(rows),'std_ddof':1}
if len(rows)==10:
    for key in ('f1_macro','auroc'):
        vals=[float(r[key]) for r in rows]
        summary[key+'_mean']=statistics.mean(vals)
        summary[key+'_std_sample']=statistics.stdev(vals)
summary['status']='COMPLETE' if len(rows)==10 else 'INCOMPLETE'
(a.formal_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (a.formal_dir/'summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
manifest={str(p.relative_to(a.formal_dir)):sha(p) for p in sorted(a.formal_dir.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
(a.formal_dir/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
