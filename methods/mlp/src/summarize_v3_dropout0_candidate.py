"""Summarize only completed v3 candidate formal-like records."""
import csv,json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/HSMAD/results/experiments/mlp/weibo/protocol_v3_dropout0');rows=sorted(csv.DictReader((R/'runs.csv').open()),key=lambda x:int(x['seed']));ok=[x for x in rows if x['status']=='OK']
if len(ok)!=10 or [int(x['seed']) for x in ok]!=list(range(10)):raise SystemExit('requires ten candidate/OK records')
f=np.array([float(x['f1_macro']) for x in ok]);a=np.array([float(x['auroc']) for x in ok]);s={'method':'MLP','dataset':'weibo','protocol':'v3_dropout0_candidate','n':10,'f1_macro_mean':float(f.mean()),'f1_macro_std_sample':float(f.std(ddof=1)),'auroc_mean':float(a.mean()),'auroc_std_sample':float(a.std(ddof=1)),'paper_f1_macro':.9009,'paper_auroc':.9119,'delta_f1_macro':float(f.mean()-.9009),'delta_auroc':float(a.mean()-.9119)}
(R/'summary.csv').write_text(','.join(s.keys())+'\n'+','.join(str(x) for x in s.values())+'\n');lines=['# MLP Weibo protocol_v3_dropout0 candidate','', '**Candidate result only; not a paper-level reproduction-success claim.**','',f"F1-Macro: {s['f1_macro_mean']:.10f} ± {s['f1_macro_std_sample']:.10f}; paper 0.9009; delta {s['delta_f1_macro']:+.10f}.",f"AUROC: {s['auroc_mean']:.10f} ± {s['auroc_std_sample']:.10f}; paper 0.9119; delta {s['delta_auroc']:+.10f}.",'','| seed | F1-Macro | AUROC | best_epoch | threshold |','| ---: | ---: | ---: | ---: | ---: |']
for x in ok:lines.append(f"| {x['seed']} | {float(x['f1_macro']):.10f} | {float(x['auroc']):.10f} | {x['best_epoch']} | {float(x['threshold']):.2f} |")
(R/'comparison.md').write_text('\n'.join(lines)+'\n');print(json.dumps(s))
