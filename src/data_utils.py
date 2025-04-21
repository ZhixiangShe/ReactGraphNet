"""
Data loading and preprocessing utilities for graph neural network node classification.
"""
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

def load_custom_data(features_path, edges_path, labels_path):
    """
    Loads feature, edge, and label data to construct a PyG Data object.

    Returns:
        X -- Features (np.ndarray)
        edge_index -- Edge index tensor (torch.LongTensor)
        y -- Label array (np.ndarray)
        label_encoder -- sklearn LabelEncoder instance
        index_to_node_id -- Original node IDs in array order
    """
    features_df = pd.read_csv(features_path)
    X = features_df.drop(columns=['node_id']).values
    index_to_node_id = features_df['node_id'].values
    node_id_to_index = {nid: i for i, nid in enumerate(index_to_node_id)}

    edges_df = pd.read_csv(edges_path)
    edge_index_directed = torch.tensor(
        [[node_id_to_index[src], node_id_to_index[tgt]] for src, tgt in zip(edges_df['source'], edges_df['target'])],
        dtype=torch.long
    ).t().contiguous()
    edge_index = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    labels_df = pd.read_csv(labels_path)
    le = LabelEncoder()
    y = le.fit_transform(labels_df['label'].values)
    return X, edge_index, y, le, index_to_node_id
