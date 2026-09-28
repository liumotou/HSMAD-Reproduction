# MLP baseline (auditable project implementation)

This directory is an isolated, reusable MLP baseline for HSMAD comparisons. It reads the existing verified dataset file and persisted frozen masks in place; it never creates data copies or new splits.

## Provenance

- Theory citation: Rosenblatt (1958), *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*.
- There is no uniquely confirmable official MLP code repository for this baseline.
- `methods/mlp/` is this project's auditable implementation, not a claimed official MLP repository.

## Weibo smoke architecture and parameters

`feature_dim -> Linear(64) -> ReLU -> Dropout(0.5) -> Linear(2)`; output is two logits. The model receives only the node feature tensor. It neither accesses graph edges nor invokes message passing or graph preprocessing.

| Item | Value | Source |
| --- | --- | --- |
| optimizer | Adam | HSMAD 论文明确规定 |
| learning rate | 0.01 | HSMAD 论文明确规定 |
| Weibo hidden dimension | 64 | HSMAD 论文明确规定 |
| layers / activation | input -> 64 -> 2; ReLU | smoke / unified baseline implementation choice |
| dropout | 0.5 | smoke / unified baseline implementation choice |
| weight decay | 1e-5 | smoke / unified baseline implementation choice |
| epoch | 5 | smoke / unified baseline implementation choice |
| patience | 100 | smoke / unified baseline implementation choice; prevents early stopping during 5 epochs |
| checkpoint / early stop | keep the first strictly best validation macro-F1 threshold; stop after `patience` non-improving validations | smoke / unified baseline implementation choice |
| threshold | select maximum validation macro-F1 from 0.05, 0.10, ..., 0.95; apply unchanged to test | 统一 baseline 实现选择 |

## Frozen Weibo formal configuration

The user approved `methods/mlp/configs/weibo_formal.json` for Weibo seeds 0–9: `max_epoch=1000`, `patience=100`, `dropout=0.5`, `weight_decay=1e-5`, no class weighting, and no sampling. These remain **本项目冻结的统一 baseline 实现选择**, not HSMAD-paper MLP hyperparameters. The three HSMAD-paper unified items remain Adam, `lr=0.01`, and Weibo `hidden_dim=64`.

## Smoke command

```bash
cd /root/autodl-tmp/HSMAD
python methods/mlp/src/train.py --dataset weibo --seed 0 --run-type smoke --config methods/mlp/configs/weibo_smoke.json
```

Formal runs are intentionally rejected by this smoke runner until a separately approved configuration is implemented.
