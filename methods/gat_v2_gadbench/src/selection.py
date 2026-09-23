"""GADBench selection contract: validation AUPRC controls stopping and checkpoint."""


def update_selection(state, epoch, validation_auprc, validation_f1_macro):
    auprc_improved = validation_auprc > state["best_auprc"]
    f1_improved = validation_f1_macro > state["best_f1"]
    if auprc_improved:
        state["best_auprc"] = validation_auprc
        state["auprc_best_epoch"] = epoch
        state["patience_counter"] = 0
    else:
        state["patience_counter"] += 1
    if f1_improved:
        state["best_f1"] = validation_f1_macro
        state["f1_best_epoch"] = epoch
    return state, auprc_improved, f1_improved
