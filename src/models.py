"""
Graph neural network model definitions.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TAGConv, ClusterGCNConv

class BaseNet(torch.nn.Module):
    """
    Base class for GNNs with two message-passing layers.
    """
    def __init__(self, conv_layer, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = conv_layer(num_features, hidden_channels)
        self.conv2 = conv_layer(hidden_channels, num_classes)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCNModel(BaseNet):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__(GCNConv, num_features, hidden_channels, num_classes)

class GATModel(BaseNet):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__(GATConv, num_features, hidden_channels, num_classes)

class SAGEModel(BaseNet):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__(SAGEConv, num_features, hidden_channels, num_classes)

class TAGModel(BaseNet):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__(TAGConv, num_features, hidden_channels, num_classes)

class ClusterGCNModel(BaseNet):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__(ClusterGCNConv, num_features, hidden_channels, num_classes)
