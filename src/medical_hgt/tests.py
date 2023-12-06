import tokenizers
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from datasets import load_dataset
import pandas as pd

from config import ROOT_DIR
from src.utils import node_types, metadata
from src.medical_hgt.dataset_builder import MedicalQADatasetBuilder
from src.medical_hgt.ml_utils import compute_llm_confidence_diff, find_subgraph_bfs


def compute_llm_relevancy_loss(batch, z_dict, conf_diff_dict):
    # Initialize loss
    loss = 0.0
    num_nodes = 0

    # Iterate over all nodes to form triplets and compute loss
    for node_type, question_conf_diff_dict in conf_diff_dict.items():
        for node_id, (question_index, conf_diff) in question_conf_diff_dict.items():
            question_representation = z_dict['question'][question_index]
            batch_node_index = torch.where(batch[node_type].node_uid == node_id)[0][0]
            current_node_representation = torch.index_select(z_dict[node_type], 0, batch_node_index)

            # Calculate the distance between the node embedding and the question node embedding
            distance = torch.norm(current_node_representation - question_representation, p=2)

            # For positive relevance, penalize being far from the question node
            # For negative relevance, penalize being close to the question node
            if conf_diff > 0:
                weighted_loss = conf_diff * distance
            elif conf_diff < 0:
                # Invert the distance measure for negative relevance
                weighted_loss = -conf_diff * (1 / (distance + 1e-6))  # adding a small constant to avoid division by zero
            else:  # relevance is around 0, neutral
                weighted_loss = 0

            # Accumulate the loss
            loss += weighted_loss

            num_nodes += 1

    return loss / num_nodes


def compute_llm_accuracy(conf_diff_dict: dict) -> float:
    confidence_diffs_list = [tup[1] for sublist in [list(d.values()) for d in list(conf_diff_dict.values())] for tup in sublist]
    diffs_tensor = torch.tensor(confidence_diffs_list)
    accuracy = torch.sum(diffs_tensor >= 0).item()
    accuracy /= len(confidence_diffs_list)

    return accuracy


class LLM(torch.nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", load_in_4bit=True, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        # self.tokenizer.use_default_system_prompt = False

    def get_predictions(self, sentence_token_ids):
        with torch.no_grad():
            outputs = self.model(sentence_token_ids)
            predictions = outputs[0]
        return predictions

    def get_confidence(self, sentence_token_ids):

        predictions = self.get_predictions(sentence_token_ids)

        # Get the next token candidates
        next_token_candidates_tensor = predictions[0, -1, :]
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

        # Return the top k candidates and their probabilities.
        return top_tokens_probabilities[correct_answer_token_index]

    def forward(self, knowledge_nodes_dict, nx_graph_data, question_dict, correct_answer):

        correct_answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        output_instructions = f'Is {correct_answer_map[correct_answer]} the correct answer? Reply only in Yes or No.'

        prompt = ("Question: {}\n"
                  "A) {}\n"
                  "B) {}\n"
                  "C) {}\n"
                  "D) {}\n"
                  "{}").format(question_dict['question'], question_dict['opa'], question_dict['opb'], question_dict['opc'], question_dict['opd'], output_instructions)

        prompt_token_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        confidence_without_context = self.get_confidence(prompt_token_ids)

        confidence_diffs_dict = {}  # a dict of dicts in the form {node_type_0: {node_index_0: conf_diff_0, node_index_1: conf_diff_1...}, ...}
        for node_type, nodes_uids in knowledge_nodes_dict.items():
            if node_type not in confidence_diffs_dict:
                confidence_diffs_dict[node_type] = {}
                for num_nodes, node_uid in enumerate(nodes_uids):
                    node_name = nx_graph_data.nodes[node_uid.item()]['name']
                    context = f'Context: The {node_type} {node_name}.\n'
                    context_token_ids = self.tokenizer.encode(context, return_tensors="pt")
                    combined_tokens = torch.cat((context_token_ids, prompt_token_ids), dim=1)
                    confidence_with_context = self.get_confidence(combined_tokens)
                    llm_confidence_diff = compute_llm_confidence_diff(confidence_without_context, confidence_with_context)
                    confidence_diffs_dict[node_type][node_uid] = llm_confidence_diff

                    if num_nodes >= 9:
                        break
        return confidence_diffs_dict


class BatchLLM(torch.nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()

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


qa_dataset = load_dataset('medmcqa')
qa_dataset = pd.DataFrame(qa_dataset['train'])

train_batches = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/test_data_01_12_23.pickle'), 'rb'))
prime_kg = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/primeKG_nx_medium.pickle'), 'rb'))


batch_llm = BatchLLM('gpt2')

for batch in train_batches:

    confidence_diffs_per_question = []
    prompt_dicts_list = []
    correct_answers_list = []
    subgraphs_list = []
    for node_index, question_node_representation in enumerate(batch['question'].x):
        qa_index = batch['question'].node_uid[node_index].item()
        subgraph_nodes_uid_dict = find_subgraph_bfs(batch, node_index, 'question')
        prompt_dict = dict(qa_dataset.iloc[qa_index].drop(['id', 'cop', 'exp']))
        correct_answer = qa_dataset.iloc[qa_index]['cop']
        subgraphs_list.append(subgraph_nodes_uid_dict)
        correct_answers_list.append(correct_answer)
        prompt_dicts_list.append(prompt_dict)

    # # Forward pass on the LLM
    # confidence_diffs_per_question = []  # a list of tuples (question_embeddings, conf_diffs_dict_per_question)
    # # compute the llm's feedback per question in the batch
    # for node_index, question_node_representation in enumerate(z_dict['question']):
    #     qa_index = batch['question'].node_uid[node_index].item()
    #     subgraph_nodes_uid_dict = find_subgraph_bfs(batch, node_index, 'question')
    #     prompt_dict = dict(qa_dataset.iloc[qa_index].drop(['id', 'cop', 'exp']))
    #     correct_answer = qa_dataset.iloc[qa_index]['cop']
    #     current_confidence_diffs_dict = llm(subgraph_nodes_uid_dict, prime_kg, prompt_dict, correct_answer)
    #     confidence_diffs_per_question.append((question_node_representation, current_confidence_diffs_dict))

    results = batch_llm(subgraphs_list, prime_kg, prompt_dicts_list, correct_answers_list)
    loss = compute_llm_relevancy_loss(batch, batch.x_dict, results)
    accuracy = compute_llm_accuracy(results)
print('end')
