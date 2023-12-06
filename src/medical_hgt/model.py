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
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", load_in_4bit=True, device_map="auto")
        # self.tokenizer.use_default_system_prompt = False
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_predictions(self, sentence_token_ids_list):
        with torch.no_grad():
            outputs = self.model(**sentence_token_ids_list)
            predictions_list = [output.unsqueeze(0) for output in outputs.logits]
        return predictions_list

    def get_confidence(self, sentence_token_ids_list):

        confidence_list = []
        predictions_list = self.get_predictions(sentence_token_ids_list)

        for prediction in predictions_list:
            # Get the next token candidates
            next_token_candidates_tensor = prediction[0, -1, :]
            next_token_candidates_tensor = next_token_candidates_tensor.to(torch.float32)

            # Get most probable tokens
            num_tokens = 1000
            most_probable_tokens_indices = torch.topk(next_token_candidates_tensor, num_tokens).indices.tolist()

            # Get all tokens' probabilities
            all_tokens_probabilities = torch.nn.functional.softmax(next_token_candidates_tensor, dim=-1)

            top_tokens_probabilities = all_tokens_probabilities[most_probable_tokens_indices].tolist()

            # Decode the top tokens back to words
            top_tokens = [self.tokenizer.decode([idx]).strip() for idx in most_probable_tokens_indices]

            correct_answer_token_index = top_tokens.index('Yes')
            wrong_answer_token_index = top_tokens.index('No')

            confidence_list.append(top_tokens_probabilities[correct_answer_token_index])

        return confidence_list

    def forward(self, knowledge_nodes_dicts_list, nx_graph_data, question_dicts_list, correct_answers_list):

        correct_answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        prompts = [("Question: {}\n"
                    "A) {}\n"
                    "B) {}\n"
                    "C) {}\n"
                    "D) {}\n"
                    "{}").format(
            question_dicts_list[i]['question'],
            question_dicts_list[i]['opa'],
            question_dicts_list[i]['opb'],
            question_dicts_list[i]['opc'],
            question_dicts_list[i]['opd'],
            f'Is {correct_answer_map[correct_answers_list[i]]} the correct answer? Reply only in Yes or No.')
            for i in range(len(correct_answers_list))]

        # Batch process all prompts (without context)
        prompt_token_ids = self.tokenizer.batch_encode_plus(prompts, padding=True, truncation=True, return_tensors="pt")
        self.unify_batch_encoding_device(prompt_token_ids)
        batch_confidence_without_context = self.get_confidence(prompt_token_ids)

        # Process contexts
        confidence_diffs_dict = {}  # a dict of dicts in the form {node_type_0: {node_index_0: conf_diff_0, node_index_1: conf_diff_1...}, ...}
        for idx, knowledge_nodes_dict in enumerate(knowledge_nodes_dicts_list):
            # We batch process per node type
            for node_type, nodes_uids in knowledge_nodes_dict.items():
                question_with_contexts = []
                node_ids = []
                if node_type not in confidence_diffs_dict:
                    confidence_diffs_dict[node_type] = {}

                for node_uid in nodes_uids:
                    # Create Context string
                    node_name = nx_graph_data.nodes[node_uid.item()]['name']
                    context = f'Context: The {node_type} {node_name}.\n'

                    question_with_contexts.append(context + prompts[idx])
                    node_ids.append(node_uid)

                # Batch process questions with contexts
                batch_question_with_context_ids = self.tokenizer.batch_encode_plus(question_with_contexts, padding=True, truncation=True, return_tensors="pt")
                self.unify_batch_encoding_device(batch_question_with_context_ids)
                question_confidence_with_context = self.get_confidence(batch_question_with_context_ids)

                # Compute confidence diffs
                question_confidence_diffs_list = compute_llm_confidence_diff(batch_confidence_without_context[idx], question_confidence_with_context)
                for index, conf_diff in enumerate(question_confidence_diffs_list):
                    confidence_diffs_dict[node_type][node_ids[index].item()] = (idx, conf_diff)

        return confidence_diffs_dict

    def unify_batch_encoding_device(self, batch_encoding):
        for t in batch_encoding:
            if torch.is_tensor(batch_encoding[t]):
                batch_encoding[t] = batch_encoding[t].to(self.model.device)

