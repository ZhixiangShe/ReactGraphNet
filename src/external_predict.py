"""
Predict on external datasets and save evaluation results.
"""
import os
import pandas as pd
import torch
import joblib
from torch_geometric.data import Data
from .metrics_utils import evaluate_model, compute_roc_pr_curve

def remap_tagconv_keys(old_state_dict):
    """
    Remap PyG TAGConv state_dict keys for compatibility.
    """
    new_state_dict = {}
    for key in old_state_dict.keys():
        if 'lin_out.weight' in key:
            new_key = key.replace('lin_out', 'lins.0')
        elif 'lin_root.weight' in key:
            new_key = key.replace('lin_root', 'lins.1')
        elif 'lin_out.bias' in key:
            new_key = key.replace('lin_out', 'lins.0')
        else:
            new_key = key
        new_state_dict[new_key] = old_state_dict[key]
    return new_state_dict

def predict_external_data(model_path, scaler_path, le_path, features_path, edges_path, labels_path, output_path,
                         ModelClass, best_params, num_features, num_classes, device):
    """
    Loads and applies the selected model and preprocessing to external data.
    Saves prediction results and metrics.
    """
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    model = ModelClass(num_features, best_params['hidden_channels'], num_classes).to(device)
    loaded_dict = torch.load(model_path, map_location=device)
    if 'TAG' in ModelClass.__name__:
        loaded_dict = remap_tagconv_keys(loaded_dict)
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()

    features_df = pd.read_csv(features_path)
    X_new = features_df.drop(columns=['node_id']).values
    X_new_scaled = scaler.transform(X_new)
    node_id_to_index = {nid: i for i, nid in enumerate(features_df['node_id'])}
    edges_df = pd.read_csv(edges_path)
    edge_index = torch.tensor(
        [[node_id_to_index[src], node_id_to_index[tgt]] for src, tgt in zip(edges_df['source'], edges_df['target'])],
        dtype=torch.long
    ).t().contiguous()
    labels_df = pd.read_csv(labels_path)
    merged_df = features_df[['node_id']].merge(labels_df, on='node_id', how='left')
    y_new = le.transform(merged_df['label'].values)

    x_new = torch.tensor(X_new_scaled, dtype=torch.float).to(device)
    edge_index_new = edge_index.to(device)
    y_new = torch.tensor(y_new, dtype=torch.long).to(device)
    data_new = Data(x=x_new, edge_index=edge_index_new, y=y_new)
    mask = torch.ones(data_new.num_nodes, dtype=torch.bool).to(device)

    acc, precision, recall, f1, macro_auc, micro_auc, probas, true_labels, cm = evaluate_model(
        model, data_new, mask, le)
    predicted_labels = le.inverse_transform(probas.argmax(axis=1))
    results_df = pd.DataFrame({
        'node_id': features_df['node_id'],
        'true_label': le.inverse_transform(true_labels),
        'predicted_label': predicted_labels,
        **{f'prob_{cls}': probas[:, i] for i, cls in enumerate(le.classes_)}
    })
    results_df.to_csv(output_path, index=False)
    # Save metrics etc...
