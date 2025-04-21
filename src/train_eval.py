"""
Training and evaluation routines including hyperparameter search.
"""
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.optim import Adam
import torch
from .metrics_utils import evaluate_model, compute_roc_pr_curve

def train_and_evaluate(model, data, epochs, optimizer, criterion, train_mask, test_mask, le, num_classes):
    """
    Training loop, returns a dictionary of metrics for train and test sets.
    """
    results = {'train': {}, 'test': {}}
    for epoch in range(epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            pred_train = out[train_mask].max(1)[1]
            acc_train = (pred_train == data.y[train_mask]).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc_train*100:.2f}%")
    # Evaluation
    for split, mask in [('train', train_mask), ('test', test_mask)]:
        acc, prec, rec, f1, macro_auc, micro_auc, probas, true, cm = evaluate_model(model, data, mask, le)
        roc_data, pr_data = compute_roc_pr_curve(true, probas, num_classes)
        results[split] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
            'macro_auc': macro_auc, 'micro_auc': micro_auc,
            'confusion_matrix': cm,
            'roc': roc_data, 'pr': pr_data
        }
    return results
