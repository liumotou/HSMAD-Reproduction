import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/experiments/gcn/amazon/protocol_v1_paper_hidden64/formal';P={'f1_macro':.6396,'auroc':.8004}
rows=[]
for s in range(10):
 r=next(csv.DictReader((OUT/f'seed_{s}'/'runs.csv').open()))
 if r['run_type']!='formal' or r['status']!='OK':raise RuntimeError(f'seed {s}: {r["status"]}')
 rows.append(r)
z={'method':'GCN','protocol_version':'v1_paper_hidden64','dataset':'amazon','run_type':'formal','n':10}
for k in P:
 v=np.array([float(r[k]) for r in rows]);z[k+'_mean']=float(v.mean());z[k+'_std_sample']=float(v.std(ddof=1));z[k+'_paper']=P[k];z[k+'_delta']=float(v.mean()-P[k])
with (OUT/'summary.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(z));w.writeheader();w.writerow(z)
lines=['# GCN Amazon protocol_v1_paper_hidden64','', '| metric | mean | sample std | paper | delta |','|---|---:|---:|---:|---:|']
for k in P:lines.append(f'| {k} | {z[k+"_mean"]:.10f} | {z[k+"_std_sample"]:.10f} | {P[k]:.4f} | {z[k+"_delta"]:+.10f} |')
(OUT/'comparison.md').write_text('\n'.join(lines)+'\n');print(json.dumps(z))
