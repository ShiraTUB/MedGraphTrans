import torch
import torch.nn.functional as F

from torch_geometric.nn import Linear
from torch_geometric.data import HeteroData

from src.utils import node_types, metadata
from weighted_hgt_conv import WeightedHGTConv


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()

        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = WeightedHGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data, edge_weights_dict):
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in data.x_dict.items()}

        # # include edges weights before passing to hgt: apply weight to all source nodes for message passing
        # weighted_x_dict = {node_type: feature_tensor.clone() for node_type, feature_tensor in x_dict.items()}
        # for edge_type, edge_indices in data.edge_index_dict.items():
        #     source_node_type = edge_type[0]
        #     source_nodes_indices = edge_indices[0]
        #
        #     # apply a sigmoid function to keep weights in (0, 1)
        #     # edge_weights[edge_type[1]].data = torch.sigmoid(edge_weights[edge_type[1]].data)
        #
        #     edge_type_weights = edge_weights[edge_type[1]].squeeze()
        #     weighted_x_dict[source_node_type][source_nodes_indices] *= edge_type_weights.unsqueeze(1).repeat(1, weighted_x_dict[source_node_type].size(1))

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, edge_weights_dict)

        return x_dict


# Our decoder applies the dot-product between source and destination node embeddings to derive edge-level predictions:
class Decoder(torch.nn.Module):
    def forward(self, x_question: torch.Tensor, x_answer: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_question = x_question[edge_label_index[0]]
        edge_feat_answer = x_answer[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return F.sigmoid((edge_feat_question * edge_feat_answer).sum(dim=-1))


class Model(torch.nn.Module):
    def __init__(self, hetero_data, hidden_channels=64):
        super().__init__()
        self.hgt = HGT(hidden_channels=hidden_channels, out_channels=64, num_heads=2, num_layers=1)
        self.decoder = Decoder()
        self.hetero_data = hetero_data

        # Initialize learnable edge weights
        self.edge_weights_dict = torch.nn.ParameterDict()

        for edge_type, edge_indices in hetero_data.edge_index_dict.items():
            edge_type = '__'.join(edge_type)
            parameter_tensor = torch.nn.Parameter(torch.randn((1, edge_indices.size(1))), requires_grad=True)
            self.edge_weights_dict[edge_type] = F.sigmoid(parameter_tensor)

    def forward(self, batch_data: HeteroData) -> (torch.Tensor, dict):
        relevant_edge_weights_dict = {}

        # find the relevant indices in the models weights dict
        for edge_type in batch_data.edge_types:
            relevant_indices = self.hetero_data[edge_type].edge_uid[batch_data[edge_type].edge_uid]
            edge_type = '__'.join(edge_type)
            relevant_edge_weights_dict[edge_type] = self.edge_weights_dict[edge_type][0][relevant_indices]

        weighted_z_dict = self.hgt(batch_data, relevant_edge_weights_dict)

        pred = self.decoder(
            weighted_z_dict["question"],
            weighted_z_dict["answer"],
            batch_data["question", "question_correct_answer", "answer"].edge_label_index,
        )

        # Make sure edges weights are between 0 and 1
        # for edge_type in self.edge_weights_dict.keys():
        #     self.edge_weights_dict[edge_type].data = self.sigmoid(self.edge_weights_dict[edge_type].data)

        return pred
