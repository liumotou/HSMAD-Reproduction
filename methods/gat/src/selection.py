"""Selection contract for full GAT diagnostics.

F1-Macro alone controls early-stopping patience.  Validation AUPRC alone
chooses the checkpoint retained for final evaluation.
"""


def update_selection(state, epoch, validation_f1_macro, validation_auprc):
    """Update independent F1-patience and AUPRC-checkpoint trackers."""
    f1_improved = validation_f1_macro > state["best_f1"]
    auprc_improved = validation_auprc > state["best_auprc"]
    if f1_improved:
        state["best_f1"] = validation_f1_macro
        state["f1_best_epoch"] = epoch
        state["patience_counter"] = 0
    else:
        state["patience_counter"] += 1
    if auprc_improved:
        state["best_auprc"] = validation_auprc
        state["checkpoint_epoch"] = epoch
    return state, f1_improved, auprc_improved
