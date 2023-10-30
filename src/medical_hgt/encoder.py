import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import HGTConv, Linear


class Encoder(nn.Module):
    def __init__(self, model_name_or_path='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        # Loads transformers model
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoded_input = None

    def forward(self, input, normalize=True, tokenize=False):
        if tokenize:
            self.tokenize(text_input=input)
            model_output = self.model(**self.encoded_input)
        else:
            self.encoded_input = input
            model_output = self.model(input)

        # if input.get_device() >= 0:
        #     attention_mask = (input > 0).detach().clone().type(torch.LongTensor).to(input.get_device())
        # else:
        #     attention_mask = (input > 0).detach().clone().type(torch.LongTensor)

        attention_mask = (input > 0).detach().clone().type(torch.LongTensor)

        sentence_embeddings = self.mean_pooling(model_output, attention_mask)

        if normalize:
            sentence_embeddings = self.normalize(sentence_embeddings)

        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def normalize(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def tokenize(self, text_input, padding=True, truncation=True, return_tensors='pt'):
        self.encoded_input = self.tokenizer(text_input, padding=True, truncation=True, return_tensors='pt')


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

        return x_dict['message']


class MedicalEncoder(torch.nn.Module):
    def __init__(self, d_model, n_types, meta):
        super().__init__()
        self.node_encoder = Encoder()
        self.graph_encoder = HGT(hidden_channels=d_model, out_channels=1, num_heads=6, num_layers=2, n_types=n_types, meta=meta)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {node_type: self.node_encoder(x) for node_type, x in x_dict.items()}

        return self.graph_encoder(x_dict, edge_index_dict)
