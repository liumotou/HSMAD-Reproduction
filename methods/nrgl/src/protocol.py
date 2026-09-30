"""Non-negotiable split and label-use contract for the NRGL candidate."""


def candidate_protocol_contract():
    return {
        "positioning": "candidate_protocol_not_author_exact",
        "frozen_masks_required": True,
        "synthetic_label_noise_injection": False,
        "loss_mask": "train_mask",
        "checkpoint_mask": "val_mask",
        "threshold_mask": "val_mask",
        "test_metric_mask": "test_mask",
    }
