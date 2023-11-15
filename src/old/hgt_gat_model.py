import torch
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from src.utils import relation_types as all_possible_edge_types
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F


class HeterogeneousGATLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # Define a GATConv layer for each edge type
        self.convs = torch.nn.ModuleDict({
            edge_type: GATConv((-1, -1), hidden_dim, heads=num_heads, concat=False)
            for edge_type in all_possible_edge_types
        })

    def forward(self, x_dict, edge_index_dict):
        out = {}
        for edge_type, conv in self.convs.items():
            src_type, rel_type, dst_type = edge_type.split('_')
            edge_index = edge_index_dict[(src_type, rel_type, dst_type)]
            out[dst_type] = conv(x_dict[src_type], x_dict[dst_type], edge_index)
        return out


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])


class HeteroGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_dim=1536, num_answer_choices=4, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Heterogeneous GAT layer
        self.hetero_gat = HeterogeneousGATLayer(hidden_dim, num_heads)

        # Output layer for answer prediction
        self.answer_out = torch.nn.Linear(hidden_dim, num_answer_choices)

        # Initialize edge weights
        # self.edge_weights = torch.nn.ParameterDict({
        #     'question_to_answer': torch.nn.Parameter(torch.ones(num_answer_choices)),
        #     'answer_to_knowledge': torch.nn.Parameter(torch.ones(num_knowledge_nodes)),
        #     # Initialize other edge weights if necessary
        # })

        self.edge_weights = torch.nn.ParameterDict({
            edge_type: torch.nn.Parameter(torch.ones(1))
            for edge_type in all_possible_edge_types
        })

    def forward(self, hetero_data):
        # Get the initial node embeddings
        x_dict = {node_type: data.x for node_type, data in hetero_data.items() if 'x' in data}

        # Prepare edge weight dictionary
        edge_weight_dict = {
            edge_type: self.edge_weights[edge_type]
            for edge_type in hetero_data.edge_index_dict.keys()
        }

        # Pass through heterogeneous GAT layer
        x_dict = self.hetero_gat(x_dict, hetero_data.edge_index_dict, edge_weight_dict)

        # Answer prediction
        answer_features = torch.cat([x_dict['question'], x_dict['answer']], dim=1)
        answer_preds = self.answer_out(answer_features)

        return answer_preds, edge_weight_dict
