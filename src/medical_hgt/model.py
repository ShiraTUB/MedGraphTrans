import torch

import torch.nn.functional as F

from torch_geometric.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import HGTConv
from typing import Tuple

from src.utils import node_types, metadata


class HGT(torch.nn.Module):
    def __init__(self, channels, num_heads, num_layers, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.lin_dict = torch.nn.ModuleDict()
        self.bn_dict = torch.nn.ModuleDict()  # Adding batch normalization layers

        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, channels)
            self.bn_dict[node_type] = torch.nn.BatchNorm1d(channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(channels, channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

    def forward(self, data):
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in data.x_dict.items()}

        if data.x_dict['question'].size(0) == self.batch_size:
             x_dict = {node_type: self.bn_dict[node_type](x) for node_type, x in x_dict.items()}  # Apply batch normalization

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

        return x_dict


# Our decoder applies the dot-product between source and destination node embeddings to derive edge-level predictions:
class Decoder(torch.nn.Module):
    def forward(self, x_question: torch.Tensor, x_answer: torch.Tensor, pos_edge_label_index: torch.Tensor, neg_edge_label_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Our decoder applies the dot-product between source and destination node embeddings to derive edge-level predictions:

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

        # pos_pred = F.sigmoid((pos_edge_feat_question * pos_edge_feat_answer).sum(dim=-1))
        pos_pred = (pos_edge_feat_question * pos_edge_feat_answer).sum(dim=-1)

        if pos_pred.dim() == 0:
            pos_pred = pos_pred.view(1)

        neg_edge_feat_question = x_question[neg_edge_label_index[0]]
        neg_edge_feat_answer = x_answer[neg_edge_label_index[1]]

        # neg_pred = F.sigmoid((neg_edge_feat_question * neg_edge_feat_answer).sum(dim=-1))
        neg_pred = (neg_edge_feat_question * neg_edge_feat_answer).sum(dim=-1)

        if neg_pred.dim() == 0:
            neg_pred = neg_pred.view(1)

        return pos_pred, neg_pred


class MedicalHGT(torch.nn.Module):
    def __init__(self, channels=64, num_heads=2, num_layers=1, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.hgt = HGT(channels=channels, num_heads=num_heads, num_layers=num_layers, batch_size=self.batch_size)
        self.decoder = Decoder()
        # self.grads = {}  # for debugging purposes

    def forward(self, batch_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z_dict = self.hgt(batch_data)
        # self.grads = {}  # for debugging purposes
        # for node_type in z_dict.keys():
        #     if z_dict[node_type].requires_grad:
        #         z_dict[node_type].register_hook(self.save_grad(node_type))  # for debugging purposes

        pos_pred, neg_pred = self.decoder(
            z_dict["question"],
            z_dict["answer"],
            batch_data["question", "question_correct_answer", "answer"].edge_label_index,
            batch_data["question", "question_wrong_answer", "answer"].edge_label_index,
        )

        return pos_pred, neg_pred, z_dict

    # def save_grad(self, name):
    #     def hook(grad):
    #         self.grads[name] = grad
    #
    #     return hook
