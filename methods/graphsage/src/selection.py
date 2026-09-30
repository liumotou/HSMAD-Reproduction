"""GADBench-style validation-AUPRC selection and patience control."""


def update_auprc_selection(state, epoch, validation_auprc, patience):
    improved = validation_auprc > state["best_auprc"]
    if improved:
        state["best_auprc"] = float(validation_auprc)
        state["best_epoch"] = int(epoch)
        state["patience_counter"] = 0
    else:
        state["patience_counter"] += 1
    return state, improved, state["patience_counter"] > patience
