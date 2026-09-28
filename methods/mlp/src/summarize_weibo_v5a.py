import csv,json,os
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/HSMAD');R=ROOT/os.environ.get('MLP_V5A_FORMAL_ROOT','results/experiments/mlp/weibo/protocol_v5a_f1_earlystop_auprc_checkpoint/formal');rows=[json.loads((R/f'seed_{s}/metrics.json').read_text()) for s in range(10)]
if not all(x['status']=='OK' and x['run_type']=='formal' and x['execution_protocol']=='project-unified-v5a' for x in rows):raise SystemExit('requires 10 formal v5a OK')
fields=[]
for x in rows:
 for k in x:
  if k not in fields:fields.append(k)
with (R/'runs.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
f=np.array([x['f1_macro'] for x in rows]);a=np.array([x['auroc'] for x in rows]);s={'method':'MLP','dataset':'weibo','protocol':'v5a_f1_earlystop_auprc_checkpoint','source_reference':'GADBench commit f9aa021ce9b6c6580427fb633b596843be76ddc6','execution_protocol':'project-unified-v5a','not_claimed_as':'byte-identical GADBench / author-exact baseline','n':10,'f1_macro_mean':float(f.mean()),'f1_macro_std_sample':float(f.std(ddof=1)),'auroc_mean':float(a.mean()),'auroc_std_sample':float(a.std(ddof=1)),'paper_f1_macro':.9009,'paper_auroc':.9119,'delta_f1_macro':float(f.mean()-.9009),'delta_auroc':float(a.mean()-.9119)}
with (R/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(s));w.writeheader();w.writerow(s)
(R/'comparison.md').write_text('# MLP Weibo v5a\n\nProject-unified-v5a; not claimed as byte-identical GADBench or author-exact baseline. Existing Weibo v3 remains preserved.\n\n'+json.dumps(s,indent=2)+'\n')
registry=ROOT/'results/experiments/mlp/experiment_protocol_registry.csv';entry={'method':'MLP','dataset':'weibo','protocol':'v5a_f1_earlystop_auprc_checkpoint','source_reference':s['source_reference'],'execution_protocol':s['execution_protocol'],'not_claimed_as':s['not_claimed_as'],'summary_path':str(R/'summary.csv'),'summary_sha256':__import__('hashlib').sha256((R/'summary.csv').read_bytes()).hexdigest()}
with registry.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(entry));w.writeheader();w.writerow(entry)
print(json.dumps(s))
