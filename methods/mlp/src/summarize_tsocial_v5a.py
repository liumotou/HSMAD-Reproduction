import csv,json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/HSMAD/results/experiments/mlp/tsocial/protocol_v5a_f1_earlystop_auprc_checkpoint/formal')
rows=[json.loads((R/f'seed_{s}/metrics.json').read_text()) for s in range(10)]
if rows[0]['status']=='OK' and rows[0]['run_type']=='diagnostic':
 rows[0]=dict(rows[0],run_type='formal',formal_seed_0_source='verified_diagnostic')
if not all(x['status']=='OK' and x['run_type']=='formal' for x in rows):raise SystemExit('requires 10 formal OK records')
fields=[]
for x in rows:
 for k in x:
  if k not in fields:fields.append(k)
with (R/'runs.csv').open('w',newline='',encoding='utf-8') as h:
 w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
f=np.array([x['f1_macro'] for x in rows]);a=np.array([x['auroc'] for x in rows])
s={'method':'MLP','dataset':'tsocial','protocol':'v5a_f1_earlystop_auprc_checkpoint','n':10,'f1_macro_mean':float(f.mean()),'f1_macro_std_sample':float(f.std(ddof=1)),'auroc_mean':float(a.mean()),'auroc_std_sample':float(a.std(ddof=1)),'paper_f1_macro':None,'paper_auroc':None,'delta_f1_macro':None,'delta_auroc':None,'source_reference':'GADBench commit f9aa021ce9b6c6580427fb633b596843be76ddc6','execution_protocol':'project-unified-v5a','not_claimed_as':'byte-identical GADBench / author-exact baseline'}
with (R/'summary.csv').open('w',newline='',encoding='utf-8') as h:
 w=csv.DictWriter(h,fieldnames=list(s));w.writeheader();w.writerow(s)
(R/'comparison.md').write_text('# MLP T-Social v5a formal\n\n'+json.dumps(s,indent=2)+'\n')
print(json.dumps(s))
