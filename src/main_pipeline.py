"""
Main pipeline script that orchestrates all steps.
"""
import os
import numpy as np
import torch
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from data_utils import load_custom_data
from models import GCNModel, GATModel, SAGEModel, TAGModel, ClusterGCNModel
from train_eval import train_and_evaluate
import joblib

# Parameter setup
DATA_DIR = '../data'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading
features_fp = os.path.join(DATA_DIR, 'features.csv')
edges_fp = os.path.join(DATA_DIR, 'edges.csv')
labels_fp = os.path.join(DATA_DIR, 'labels.csv')
X, edge_index, y, le, index_to_node_id = load_custom_data(features_fp, edges_fp, labels_fp)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)
num_nodes, num_features = x.shape
num_classes = len(le.classes_)

# Train-test split and PyG Data
indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(indices, stratify=y, test_size=0.2, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask).to(DEVICE)

# Model loop and save results
param_grid = {
    'hidden_channels': [16, 32, 64],
    'lr': [0.01, 0.02, 0.05],
    'weight_decay': [5e-4, 1e-3],
    'epochs': [100, 200]
}
models = [GCNModel, GATModel, SAGEModel, TAGModel, ClusterGCNModel]
# ...hyperparameter search using train_and_evaluate, save as in above integration code...

# Save best model, scaler and encoder
torch.save(best_model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler.pkl'))
joblib.dump(le, os.path.join(RESULTS_DIR, 'label_encoder.pkl'))

# ...call explainability.py's explain_node and external_predict.py's predict_external_data if needed...
