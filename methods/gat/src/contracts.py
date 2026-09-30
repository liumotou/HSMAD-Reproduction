"""Immutable contract for the selected original-GAT Weibo smoke candidate."""


def selected_candidate_contract():
    """Return the auditable architecture properties approved for Candidate A."""
    return {
        "hidden_heads": 8,
        "hidden_features_per_head": 8,
        "hidden_concat_dim": 64,
        "hidden_activation": "ELU",
        "output_heads": 1,
        "output_dim": 2,
        "output_concat": False,
        "attention_negative_slope": 0.2,
    }
