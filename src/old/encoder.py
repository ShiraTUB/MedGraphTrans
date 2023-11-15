import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv, Linear


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedded_input, apply_normalization=True):

        if embedded_input.get_device() >= 0:
            attention_mask = (embedded_input > 0).detach().clone().type(torch.LongTensor).to(embedded_input.get_device())
        else:
            attention_mask = (embedded_input > 0).detach().clone().type(torch.LongTensor)

        sentence_embeddings = self.mean_pooling(embedded_input, attention_mask)

        if apply_normalization:
            sentence_embeddings = self.normalize(sentence_embeddings)

        return sentence_embeddings

    def mean_pooling(self, embedded_input, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embedded_input.size()).float()
        return torch.sum(embedded_input * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def normalize(self, sentence_embeddings):
        output = F.normalize(sentence_embeddings, p=2, dim=1)
        return output


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, n_types, meta):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()

        for node_type in n_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, meta, num_heads, group='sum')
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # TODO: is answer the right thing to return?
        return x_dict['answer']


class MedicalEncoder(torch.nn.Module):
    def __init__(self, d_model, n_types, meta):
        super().__init__()
        self.node_encoder = Encoder()
        self.graph_encoder = HGT(hidden_channels=d_model, out_channels=1, num_heads=6, num_layers=2, n_types=n_types, meta=meta)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {node_type: self.node_encoder(x) for node_type, x in x_dict.items()}

        return self.graph_encoder(x_dict, edge_index_dict)
