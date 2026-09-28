import csv,json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/HSMAD/results/experiments/mlp/amazon/protocol_v5_val_auprc_selection/formal');rows=[]
for seed in range(10):rows.append(json.loads((R/f'seed_{seed}/metrics.json').read_text()))
if not all(x['status']=='OK' and x['run_type']=='formal' for x in rows):raise SystemExit('requires 10 formal OK')
fields=[]
for row in rows:
 for key in row:
  if key not in fields:fields.append(key)
with (R/'runs.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
f=np.array([x['f1_macro'] for x in rows]);a=np.array([x['auroc'] for x in rows]);s={'method':'MLP','dataset':'amazon','protocol':'v5_val_auprc_selection','n':10,'f1_macro_mean':float(f.mean()),'f1_macro_std_sample':float(f.std(ddof=1)),'auroc_mean':float(a.mean()),'auroc_std_sample':float(a.std(ddof=1)),'paper_f1_macro':.9223,'paper_auroc':.9801,'delta_f1_macro':float(f.mean()-.9223),'delta_auroc':float(a.mean()-.9801)}
with (R/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(s));w.writeheader();w.writerow(s)
(R/'comparison.md').write_text('# MLP Amazon v5 validation-AUPRC selection\n\nIndependent protocol; v3 is retained as validation-F1 checkpoint protocol, not the current primary candidate.\n\n'+json.dumps(s,indent=2)+'\n');print(json.dumps(s))
