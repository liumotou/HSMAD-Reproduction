"""Formal summary for isolated GraphConsis candidate artifacts."""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np


def sample_summary(rows):
    f1=np.asarray([float(r["f1_macro"]) for r in rows]); auc=np.asarray([float(r["auroc"]) for r in rows])
    return {"n":len(rows),"f1_macro_mean":float(f1.mean()),"f1_macro_std_sample":float(f1.std(ddof=1)),
            "auroc_mean":float(auc.mean()),"auroc_std_sample":float(auc.std(ddof=1))}


def sha256_file(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(formal: Path):
    rows=[]; audits={}
    for seed in range(10):
        metrics_path=formal/f"seed_{seed}"/"metrics.json"
        audit_path=formal/f"seed_{seed}"/"audit_recompute/recompute_result.json"
        metrics=json.loads(metrics_path.read_text()); audit=json.loads(audit_path.read_text())
        if metrics["run_type"]!="formal" or metrics["status"]!="OK" or audit["status"]!="recompute_match":
            raise RuntimeError(f"seed {seed} is not eligible")
        rows.append(metrics); audits[f"seed_{seed}"]=sha256_file(audit_path)
    result=sample_summary(rows); result.update({"method":"GraphConsis","dataset":rows[0]["dataset"],
        "protocol_version":rows[0]["protocol_version"],"candidate_protocol_not_author_exact":True,
        "eligibility":"formal_OK_and_checkpoint_recompute_match_only"})
    fields=["seed","f1_macro","auroc","best_epoch","threshold","actual_epochs","peak_gpu_mb","wall_time_sec","predicted_anomaly_count"]
    with (formal/"seed_level.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in rows])
    with (formal/"summary.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(result)); w.writeheader(); w.writerow(result)
    (formal/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    (formal/"formal_audit.json").write_text(json.dumps({"status":"FORMAL_CANDIDATE_SUMMARY","summary":result,"recompute_audit_sha256":audits},indent=2,sort_keys=True))
    files=[p for p in formal.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"]
    (formal/"artifact_manifest.json").write_text(json.dumps({str(p.relative_to(formal)):sha256_file(p) for p in sorted(files)},indent=2,sort_keys=True))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--formal",required=True); a=p.parse_args(); print(json.dumps(summarize(Path(a.formal)),sort_keys=True))


if __name__=="__main__": main()
