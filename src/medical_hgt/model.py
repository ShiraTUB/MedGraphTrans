import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import HGTConv, Linear
from transformers import BertTokenizer, BertModel


# 1. Define a BERT-based node encoder
class NodeEncoder(torch.nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super(NodeEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.bert(**inputs)
        return outputs['pooler_output']  # using [CLS] token embedding as node embedding


# 2. Define the HGT model
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, n_types, meta):
        super(HGT, self).__init__()

        self.lin_dict = torch.nn.ModuleDict()

        for node_type in n_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, meta, num_heads, group='sum')
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_type):
        return self.conv(x, edge_index, edge_type)


class MedicalQAModel(pl.LightningModule):
    def __init__(self, encoder, decoder, tgt_emb, generator, lr=3e-4, optimizer=torch.optim.AdamW):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.optimizer = optimizer
        self.learning_rate = lr
        self.validation_step_outputs = []
        self.model = HGT(768, 128, 2)

    def forward(self, questions):
        encodings = self.encoder(x_dict, edge_index_dict)
        return self.decoder(self.tgt_emb(tgt), encodings, src_mask=src_mask, tgt_mask=tgt_mask)

    def score_pairs(self, question_emb, answer_embs):
        # Implement a scoring function.
        # For simplicity, using dot product. You can use more complex functions.
        scores = torch.matmul(question_emb, answer_embs.t())
        return scores

    def training_step(self, batch, batch_idx):
        # batch contains question nodes, answer nodes, and labels indicating the correct answer
        question_nodes = batch.question_nodes
        answer_nodes = batch.answer_nodes  # This can be a batch of answers per question
        labels = batch.labels  # Correct answers

        question_embs = self.model.node_embeddings(question_nodes)
        answer_embs = self.model.node_embeddings(answer_nodes)

        scores = self.model.score_pairs(question_embs, answer_embs)

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(scores, labels)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
