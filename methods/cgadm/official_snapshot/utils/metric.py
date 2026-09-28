import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import time


@torch.no_grad()
def anomaly_evaluate(model, Data, device, opt):
    def eval(labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            labels = np.array(labels)
            k = labels.sum()
        score['RecK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
        return score


    model.eval()

    start_time = time.time()
    probs = model.predict()
    print("Time for prediction: ", time.time() - start_time)

    val_score = eval(Data.val_labels, probs[Data.graph.val_masks])
    test_score = eval(Data.test_labels, probs[Data.graph.test_masks])

    return val_score, test_score

