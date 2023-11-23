import torch
import torch.nn.functional as F

from torch_geometric.nn import Linear
from torch_geometric.data import HeteroData

from src.utils import node_types, metadata
from src.medical_hgt.weighted_hgt_conv import WeightedHGTConv


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
    def forward(self, x_question: torch.Tensor, x_answer: torch.Tensor, pos_edge_label_index: torch.Tensor, neg_edge_label_index: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
        x_question (torch.Tensor): Embeddings of 'question' nodes.
        x_answer (torch.Tensor): Embeddings of 'answer' nodes.
        pos_edge_label_index (torch.Tensor): Indices of positive edges (edges that exist).
        neg_edge_label_index (torch.Tensor): Indices of negative edges (edges that do not exist).

        Returns:
        tuple: A tuple containing two tensors (pos_pred, neg_pred) representing the predicted probabilities
               for positive and negative edges, respectively.
        """

        # Convert node embeddings to edge-level representations:
        pos_edge_feat_question = x_question[pos_edge_label_index[0]]
        pos_edge_feat_answer = x_answer[pos_edge_label_index[1]]

        pos_pred = F.sigmoid((pos_edge_feat_question * pos_edge_feat_answer).sum(dim=-1))

        if pos_pred.dim() == 0:
            pos_pred = pos_pred.view(1)

        neg_edge_feat_question = x_question[neg_edge_label_index[0]]
        neg_edge_feat_answer = x_answer[neg_edge_label_index[1]]

        neg_pred = F.sigmoid((neg_edge_feat_question * neg_edge_feat_answer).sum(dim=-1))

        if neg_pred.dim() == 0:
            neg_pred = neg_pred.view(1)

        return pos_pred, neg_pred


class Model(torch.nn.Module):
    def __init__(self, all_edges_dict, hidden_channels=64):
        super().__init__()
        self.hgt = HGT(hidden_channels=hidden_channels, out_channels=64, num_heads=2, num_layers=1)
        self.decoder = Decoder()
        # self.edge_weights_dict = self.hgt.edge_weights_dict
        # self.relevant_weights_dict = self.hgt.relevant_weights_dict
        # Initialize learnable edge weights
        self.edge_weights_dict = {}
        self.relevant_edge_indices_dict = {}
        self.selected_weights_dict = torch.nn.ParameterDict()
        self.all_edges_dict = all_edges_dict

        for edge_type, edge_indices in all_edges_dict.items():
            edge_type = '__'.join(edge_type)
            parameter_tensor = F.sigmoid(torch.randn(len(edge_indices)))
            self.edge_weights_dict[edge_type] = parameter_tensor

    def forward(self, batch_data: HeteroData) -> (torch.Tensor, torch.Tensor):

        # relevant_weights_dict = {}

        # find the relevant indices in the models weights dict
        for edge_type in batch_data.edge_types:
            if hasattr(batch_data[edge_type], "edge_index_uid"):
                relevant_indices = torch.tensor([torch.where(self.all_edges_dict[edge_type] == x)[0] for x in batch_data[edge_type].edge_index_uid])
            else:
                relevant_indices = torch.tensor([torch.where(self.all_edges_dict[edge_type] == x)[0] for x in batch_data[edge_type].edge_uid])

            edge_type = '__'.join(edge_type)
            self.relevant_edge_indices_dict[edge_type] = relevant_indices

            selected_edge_weight = torch.index_select(
                self.edge_weights_dict[edge_type], 0, relevant_indices
            )

            self.selected_weights_dict[edge_type] = torch.nn.Parameter(selected_edge_weight, requires_grad=True)

        weighted_z_dict = self.hgt(batch_data, self.selected_weights_dict)

        pos_pred, neg_pred = self.decoder(
            weighted_z_dict["question"],
            weighted_z_dict["answer"],
            batch_data["question", "question_correct_answer", "answer"].edge_label_index,
            batch_data["question", "question_wrong_answer", "answer"].edge_label_index,
        )

        return pos_pred, neg_pred
