import torch

import torch.nn.functional as F

from torch_geometric.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import HGTConv
from transformers import AutoTokenizer, AutoModel
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import node_types, metadata
from src.medical_hgt.ml_utils import compute_llm_confidence_diff, query_chatbot


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()

        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in data.x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

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


class MedicalHGT(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.hgt = HGT(hidden_channels=hidden_channels, out_channels=64, num_heads=2, num_layers=1)
        self.decoder = Decoder()
        self.grads = {}  # for debugging purposes

    def forward(self, batch_data: HeteroData) -> (torch.Tensor, torch.Tensor, dict):

        self.grads = {}  # for debugging purposes
        z_dict = self.hgt(batch_data)
        for node_type in z_dict.keys():
            if z_dict[node_type].requires_grad:
                z_dict[node_type].register_hook(self.save_grad(node_type))  # for debugging purposes

        pos_pred, neg_pred = self.decoder(
            z_dict["question"],
            z_dict["answer"],
            batch_data["question", "question_correct_answer", "answer"].edge_label_index,
            batch_data["question", "question_wrong_answer", "answer"].edge_label_index,
        )

        return pos_pred, neg_pred, z_dict

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook


class LLM(torch.nn.Module):
    # def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
    #     super().__init__()
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, knowledge_nodes_dict, nx_graph_data, dataset_question_dict, correct_answer):

        correct_answer_dict = {0: 'opa', 1: 'opb', 2: 'opc', 3: 'opd'}
        output_instructions = f'how confident are you that the correct answer is {correct_answer_dict[correct_answer]}? Return a float between 0 and 1.'
        confidence_without_context = query_chatbot(str(dataset_question_dict), output_instructions)

        confidence_diffs_dict = {}  # a dict of dicts in the form {node_type_0: {node_index_0: conf_diff_0, node_index_1: conf_diff_1...}, ...}
        for node_type, nodes_uids in knowledge_nodes_dict.items():
            if node_type not in confidence_diffs_dict:
                confidence_diffs_dict[node_type] = {}
                for node_uid in nodes_uids:
                    node_name = nx_graph_data.nodes[node_uid.item()]['name']
                    dataset_question_dict['context'] = f'The {node_type} {node_name}.'
                    confidence_with_context = query_chatbot(str(dataset_question_dict), output_instructions)
                    llm_confidence_diff = compute_llm_confidence_diff(float(confidence_without_context), float(confidence_with_context))
                    confidence_diffs_dict[node_type][node_uid] = llm_confidence_diff

        return confidence_diffs_dict
