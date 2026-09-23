# CGADM-HSMAD adapted candidate

This directory contains a thin, auditable adapter around the fixed official
CGADM source snapshot at commit
`53b83107e940b57ea0032421ed2fdbee4c7cbed0`.

It is labelled `candidate_protocol_not_author_exact`. The official CGADM
diffusion/BWGNN model and recommended training defaults are retained, while
the data boundary is replaced by the project's verified HSMAD files,
preprocessing, frozen train/validation/test masks, seeds, validation threshold
selection, and final F1-Macro/AUROC reporting.

The adapter never regenerates masks. Training/prior fitting use only the train
mask, checkpoint and threshold selection use only validation, and test is
evaluated only after both are fixed.

