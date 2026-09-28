"""Build the isolated Amazon v3 formal aggregate, optionally admitting verified diagnostic seed 0."""
import argparse,csv,hashlib,json,os,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/HSMAD');OUT=ROOT/'results/experiments/mlp/amazon/protocol_v3_dropout0/formal';DIAG=ROOT/'results/experiments/mlp/amazon/protocol_v3_dropout0_seed0_diagnostic/seed_0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args()
 config=ROOT/'methods/mlp/configs/amazon_protocol_v3_dropout0_seed0_diagnostic.json';diag_config=DIAG/'config_snapshot.json';diag_metrics=json.loads((DIAG/'metrics.json').read_text());diag_pre=json.loads((DIAG/'preflight.json').read_text())
 now_code={p:sha(ROOT/p) for p in ['methods/mlp/src/model.py','methods/mlp/src/utils.py','methods/mlp/src/train.py']}
 checks={'diagnostic_status_ok':diag_metrics['status']=='OK','config_byte_identical':config.read_bytes()==diag_config.read_bytes(),'core_code_sha_identical':json.loads(diag_metrics['code_sha256'])==now_code,'input_sha_identical':diag_pre['input_sha256']=={k:diag_metrics[k] for k in diag_pre['input_sha256']},'mask_sha_identical':diag_pre['frozen_hsmad_masks_match'] is True,'environment_identical':diag_pre['environment']==json.loads((DIAG/'environment/framework_versions.json').read_text())}
 audit={'seed_0_source':str(DIAG),'checks':checks,'included':all(checks.values())}
 (OUT/'seed_0_inclusion_audit.json').write_text(json.dumps(audit,indent=2))
 if not audit['included']:raise SystemExit(json.dumps(audit,indent=2))
 if args.precheck_only:
  print(json.dumps(audit));return
 target=OUT/'seed_0'
 if not target.exists():os.symlink('../../protocol_v3_dropout0_seed0_diagnostic/seed_0',target)
 rows=[];first=dict(diag_metrics);first.update({'run_type':'formal','protocol_version':'v3_dropout0','formal_seed_0_source':'verified_diagnostic','formal_seed_0_inclusion_audit':'seed_0_inclusion_audit.json'});rows.append(first)
 for seed in range(1,10):rows.append(json.loads((OUT/f'seed_{seed}/metrics.json').read_text()))
 if [int(x['seed']) for x in rows]!=list(range(10)) or not all(x['status']=='OK' for x in rows):raise SystemExit('requires ten formal/OK results')
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with (OUT/'runs.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 f=np.array([float(x['f1_macro']) for x in rows]);a=np.array([float(x['auroc']) for x in rows]);s={'method':'MLP','dataset':'amazon','protocol':'v3_dropout0','n':10,'f1_macro_mean':float(f.mean()),'f1_macro_std_sample':float(f.std(ddof=1)),'auroc_mean':float(a.mean()),'auroc_std_sample':float(a.std(ddof=1)),'paper_f1_macro':.9223,'paper_auroc':.9801,'delta_f1_macro':float(f.mean()-.9223),'delta_auroc':float(a.mean()-.9801)}
 with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(s));w.writeheader();w.writerow(s)
 lines=['# MLP Amazon protocol_v3_dropout0 formal','',f"F1-Macro: {s['f1_macro_mean']:.10f} ± {s['f1_macro_std_sample']:.10f}; paper 0.9223; delta {s['delta_f1_macro']:+.10f}.",f"AUROC: {s['auroc_mean']:.10f} ± {s['auroc_std_sample']:.10f}; paper 0.9801; delta {s['delta_auroc']:+.10f}.",'','| seed | F1-Macro | AUROC | best_epoch | threshold | source |','| ---: | ---: | ---: | ---: | ---: | --- |']
 for row in rows:lines.append(f"| {row['seed']} | {float(row['f1_macro']):.10f} | {float(row['auroc']):.10f} | {row['best_epoch']} | {float(row['threshold']):.2f} | {row.get('formal_seed_0_source','formal run')} |")
 (OUT/'comparison.md').write_text('\n'.join(lines)+'\n');print(json.dumps(s))
if __name__=='__main__':main()
