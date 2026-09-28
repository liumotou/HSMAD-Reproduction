import csv,json
from pathlib import Path
from methods.project_paths import project_root
import numpy as np
R=project_root() / 'results/experiments/mlp/amazon/protocol_v5a_f1_earlystop_auprc_checkpoint/formal';rows=[json.loads((R/f'seed_{s}/metrics.json').read_text()) for s in range(10)]
if not all(x['status']=='OK' and x['run_type']=='formal' for x in rows):raise SystemExit('requires 10 formal OK')
fields=[]
for x in rows:
 for k in x:
  if k not in fields:fields.append(k)
with (R/'runs.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
f=np.array([x['f1_macro'] for x in rows]);a=np.array([x['auroc'] for x in rows]);s={'method':'MLP','dataset':'amazon','protocol':'v5a_f1_earlystop_auprc_checkpoint','n':10,'f1_macro_mean':float(f.mean()),'f1_macro_std_sample':float(f.std(ddof=1)),'auroc_mean':float(a.mean()),'auroc_std_sample':float(a.std(ddof=1)),'paper_f1_macro':.9223,'paper_auroc':.9801,'delta_f1_macro':float(f.mean()-.9223),'delta_auroc':float(a.mean()-.9801)}
with (R/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(s));w.writeheader();w.writerow(s)
(R/'comparison.md').write_text('# MLP Amazon v5a\n\nv3 is retained unchanged as validation-F1 checkpoint protocol, not the current primary candidate.\n\n'+json.dumps(s,indent=2)+'\n');print(json.dumps(s))
