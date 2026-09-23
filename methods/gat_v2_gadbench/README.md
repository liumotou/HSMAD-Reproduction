# GAT-v2-GADBench — design only

Status: **DESIGN_FROZEN_NOT_EXECUTED**. This directory contains no training
code, data copy, smoke output, formal output, checkpoint, or runs CSV.

The recommended future protocol is documented in `audit/design_audit.md` and
its non-executable configuration draft is in `configs/`.

All future implementation code must remain in `methods/gat_v2_gadbench/src/`.
All future experiment outputs must remain in
`results/experiments/gat_v2_gadbench/`. Neither path may mix with the archived
`methods/gat/` implementation or `results/experiments/gat/` outputs.

## T-Social hidden-10 smoke

The isolated T-Social smoke configuration is documented in
`audit/tsocial_hidden10_smoke_design.md`. Its total hidden width is 10 as
required by HSMAD paper §5.1; its 2 heads × 5 dimensions is a project GAT-v2
structural choice, not a claimed author-published parameter.
