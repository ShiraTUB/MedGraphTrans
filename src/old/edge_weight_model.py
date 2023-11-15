import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F


class EdgeWeightNetwork(torch.nn.Module):
    def __init__(self, node_feature_dim):
        super().__init__()
        # A simple linear layer to transform concatenated node features to a single edge weight
        self.linear = torch.nn.Linear(2 * node_feature_dim, 1)

    def forward(self, node_features, edge_index):
        # Extract features of the source and target nodes
        source_node_features = node_features[edge_index[0]]
        target_node_features = node_features[edge_index[1]]

        # Concatenate features and pass through the network
        edge_features = torch.cat([source_node_features, target_node_features], dim=1)
        return torch.sigmoid(self.linear(edge_features))  # Use sigmoid to keep weights between 0 and 1
