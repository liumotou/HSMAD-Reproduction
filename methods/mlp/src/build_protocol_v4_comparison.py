"""Build a four-cell v1/v2/v3/v4 factor comparison; no training."""
import json
from pathlib import Path

R = Path('/root/autodl-tmp/HSMAD')
P = R / 'results/experiments/mlp/weibo/protocol_v4_class_weight_dropout_zero_probe'
v1 = json.loads((R / 'results/experiments/mlp/weibo/formal/seed_0/metrics.json').read_text())
v2 = json.loads((R / 'results/experiments/mlp/weibo/protocol_v2_class_weight_probe/seed_0/metrics.json').read_text())
v3 = json.loads((R / 'results/experiments/mlp/weibo/protocol_v3_dropout_zero_probe/seed_0/metrics.json').read_text())
v4 = json.loads((P / 'seed_0/metrics.json').read_text())
audit = json.loads((R / 'audit/mlp_v1_recompute.json').read_text())
t1 = next(x for x in audit['runs'] if x['seed'] == 0)['splits']['test_mask']
pre = json.loads((P / 'seed_0/preflight.json').read_text())

def info(name, metric, dropout, weight, legacy=None):
    cm = legacy['confusion_matrix_labels_0_1'] if legacy else metric['test_confusion_matrix_labels_0_1']
    count = sum(cm[i][1] for i in (0, 1)) if legacy else metric['test_predicted_anomaly_count']
    ratio = legacy['predicted_anomaly_ratio'] if legacy else metric['test_predicted_anomaly_ratio']
    return {'protocol': name, 'f1': metric['f1_macro'], 'auc': metric['auroc'], 'epoch': metric['best_epoch'],
            'threshold': metric['threshold'], 'n': count, 'ratio': ratio, 'cm': cm, 'dropout': dropout,
            'class_weight': weight, 'paper_f1_delta': metric['f1_macro'] - .9223,
            'paper_auc_delta': metric['auroc'] - .9801}

rows = [info('v1', v1, .5, 'none', t1), info('v2', v2, .5, f"[1.0, {v2['class_weight_anomaly']}]"),
        info('v3', v3, 0., 'none'), info('v4', v4, 0., f"[1.0, {v4['class_weight_anomaly']}]")]

def delta(first, second):
    return {'f1': second['f1'] - first['f1'], 'auc': second['auc'] - first['auc']}

effects = {
    'class_weight_at_dropout_0_5': delta(rows[0], rows[1]),
    'class_weight_at_dropout_0_0': delta(rows[2], rows[3]),
    'dropout_0_5_to_0_0_no_weight': delta(rows[0], rows[2]),
    'dropout_0_5_to_0_0_with_weight': delta(rows[1], rows[3]),
}
out = {'conclusion': '类别权重与 dropout 的 2x2 单-seed 因子对照；不构成论文级复现成功声明。',
       'paper': {'f1': .9223, 'auc': .9801}, 'rows': rows, 'effects': effects, 'v1_v4_preflight': pre}
(P / 'comparison.json').write_text(json.dumps(out, indent=2))
lines = ['# MLP Weibo seed=0: v1/v2/v3/v4 2×2 factor comparison', '',
         '**Conclusion: 类别权重与 dropout 的 2×2 单-seed 因子对照。** 不构成论文级复现成功声明。', '',
         '| Protocol | F1-Macro | AUROC | best_epoch | threshold | test predicted anomalies | class_weight | dropout | ΔF1 paper | ΔAUROC paper |',
         '| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |']
for x in rows:
    lines.append(f"| {x['protocol']} | {x['f1']:.10f} | {x['auc']:.10f} | {x['epoch']} | {x['threshold']:.2f} | {x['n']} ({x['ratio']:.10f}) | {x['class_weight']} | {x['dropout']:.1f} | {x['paper_f1_delta']:+.10f} | {x['paper_auc_delta']:+.10f} |")
lines += ['', '## Test confusion matrices (labels [0,1])', '']
lines += [f"- {x['protocol']}: `{x['cm']}`" for x in rows]
lines += ['', '## Single-variable effects', '', '| Effect | ΔF1-Macro | ΔAUROC |', '| --- | ---: | ---: |']
for key, value in effects.items():
    lines.append(f"| {key} | {value['f1']:+.10f} | {value['auc']:+.10f} |")
lines += ['', '## v1 vs v4 preflight assertion', '',
          '- Permitted and observed differences only: class_weight `none -> [1.0, 8.688760806916427]`; dropout `0.5 -> 0.0`.',
          '- Shared frozen inputs, code/environment hashes, GADBench reference and edge_access=none are in `seed_0/preflight.json`.',
          '- AUROC uses `softmax(logits)[:,1]` in all probes.', '']
(P / 'comparison.md').write_text('\n'.join(lines))
print(json.dumps(out))
