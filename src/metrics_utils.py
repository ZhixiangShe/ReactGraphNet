"""
Metric computations, ROC and PR curve utilities.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
)

def evaluate_model(model, data, mask, label_encoder):
    """
    Returns accuracy, precision, recall, f1, macro/micro auc, probabilities, labels, confusion matrix.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out[mask].max(1)[1]
        true = data.y[mask]
        probas = F.softmax(out[mask], dim=1).cpu().numpy()
        acc = accuracy_score(true.cpu(), preds.cpu())
        cm = confusion_matrix(true.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            true.cpu(), preds.cpu(), average='weighted'
        )
        macro_auc = roc_auc_score(true.cpu(), probas, multi_class='ovr', average='macro')
        micro_auc = roc_auc_score(true.cpu(), probas, multi_class='ovr', average='micro')
        return acc, precision, recall, f1, macro_auc, micro_auc, probas, true.cpu().numpy(), cm

def compute_roc_pr_curve(y_true, y_proba, num_classes):
    """
    Computes ROC and PR curve data for each class.
    """
    roc_data, pr_data = [], []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        for j in range(len(fpr)):
            roc_data.append({'class': i, 'fpr': fpr[j], 'tpr': tpr[j], 'auc': roc_auc})
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_proba[:, i])
        pr_auc = auc(recall, precision)
        for j in range(len(precision)):
            pr_data.append({'class': i, 'precision': precision[j], 'recall': recall[j], 'auc': pr_auc})
    return roc_data, pr_data
