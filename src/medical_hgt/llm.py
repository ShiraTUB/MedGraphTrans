import os
import torch
import gc
import pickle

from transformers import logging as hf_logging
from config import ROOT_DIR


class LLMFeedback:

    def __init__(self, question_uid: int, response_without_context: str, confidence_without_context: float, cop_confidence_without_context: float, is_correct_without_context: bool):
        self.question_uid = question_uid
        self.response_without_context = response_without_context
        self.confidence_without_context = confidence_without_context
        self.cop_confidence_without_context = cop_confidence_without_context
        self.is_correct_without_context = is_correct_without_context
        self.confidences_with_context = {}
        self.accuracies_with_context = {}
        self.cop_confidences_with_context = {}

    def insert_feedback(self, node_type: str, node_uid: int, confidence_with_context: float, cop_confidence_wit_context: float, is_correct: bool):
        if node_type not in self.confidences_with_context:
            self.confidences_with_context[node_type] = {}
            self.accuracies_with_context[node_type] = {}
            self.cop_confidences_with_context[node_type] = {}

        self.confidences_with_context[node_type][node_uid] = confidence_with_context
        self.accuracies_with_context[node_type][node_uid] = 1 if is_correct else 0
        self.cop_confidences_with_context[node_type][node_uid] = cop_confidence_wit_context

    def print_feedback(self):
        print(f"Response without context: {self.response_without_context}")
        print(f"Confidence without context: {self.confidence_without_context}")
        print(f"Cop confidence without context: {self.cop_confidence_without_context}")
        print(f"LLM is correct without context: {self.is_correct_without_context}")
        print(f"Confidences with context: {self.confidences_with_context}")
        print(f"Accuracies with context: {self.accuracies_with_context}")
        print(f"Cop confidences with context: {self.cop_confidences_with_context}")
        print("\n")


class LLM:
    def __init__(self, model, tokenizer):

        self.model = model

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device

        self.input_encoding = None
        self.output_encoding = None
        self.predictions = None
        self.model_outputs = None

        # Set the LLM's logging level to ERROR to suppress lower level logs
        hf_logging.set_verbosity_error()

    def free_llm_memory(self):
        for tup in self.model_outputs.past_key_values:
            for t in tup:
                t = t.cpu()
                # print(t.device)

        # del self.model_outputs['scores']
        del self.model_outputs.sequences
        gc.collect()

    def free_encodings_memory(self):
        for t in self.input_encoding:
            if torch.is_tensor(self.input_encoding[t]):
                self.input_encoding[t] = self.input_encoding[t].cpu()

    def verify_encoding_device(self):
        for t in self.input_encoding:
            if torch.is_tensor(self.input_encoding[t]):
                self.input_encoding[t] = self.input_encoding[t].to(self.device)

    def inference(self, prompt):

        instruct_prompt = self.format_prompt(prompt)

        self.input_encoding = self.tokenizer(instruct_prompt, return_tensors="pt")

        self.verify_encoding_device()

        self.model_outputs = self.model.generate(self.input_encoding.input_ids,
                                                 attention_mask=self.input_encoding.attention_mask,
                                                 output_scores=True,
                                                 max_new_tokens=4,
                                                 return_dict_in_generate=True)

        self.predictions = self.model_outputs.scores
        self.output_encoding = self.model_outputs.sequences

        self.free_encodings_memory()
        self.free_llm_memory()

    def get_confidence(self, correct_answer) -> dict:

        # Get the model's prediction indices
        response_ids = self.output_encoding[:, self.input_encoding.input_ids.size(1):].squeeze(0).tolist()

        most_probable_tokens_ids = torch.topk(self.predictions[0], 100).indices.tolist()[0]

        # Get all tokens' probabilities
        all_tokens_probabilities = torch.nn.functional.softmax(self.predictions[0], dim=-1)

        response_tokens_probabilities = all_tokens_probabilities.squeeze()[response_ids].tolist()
        top_tokens_probabilities = all_tokens_probabilities.squeeze()[most_probable_tokens_ids].tolist()

        # Decode the top tokens back to words
        response_tokens = [self.tokenizer.decode(id).strip() for id in response_ids]
        top_tokens = [self.tokenizer.decode(id).strip() for id in most_probable_tokens_ids]

        if response_tokens[0] in ['A', 'B', 'C', 'D']:

            llm_is_correct = response_tokens[0] == correct_answer

            answer_probability = response_tokens_probabilities[0]

            try:
                cop_index = top_tokens.index(correct_answer)
                cop_probability = top_tokens_probabilities[cop_index]

            except Exception as e:
                cop_probability = -1

            return {'response': response_tokens[0], 'confidence': answer_probability, 'cop_confidence': cop_probability, 'accuracy': llm_is_correct}

        else:
            return {'response': None, 'confidence': -1, 'cop_confidence': -1, 'accuracy': False}

    def reset(self):
        self.input_encoding = None
        self.output_encoding = None
        self.model_outputs = None
        self.predictions = None

    def format_prompt(self, str):

        # str should have the strcture: Question: A headache could be a symptom of? A. Diabetes B. Concussion C. Broken leg D. Hemorrhoids

        instruction_beginning = "[INST] Your task is to answer multiple choice questions. You MUST respond only with a single letter of the correct answer. Question: Greenhouses are great for plants like A. Pizza B. Lollipops C. Candles D. French beans Answer: [/INST] D [INST] Your task is to answer multiple choice questions. You MUST respond only with a single letter of the correct answer. "
        instruction_end = " Answer: [/INST]"

        return instruction_beginning + str + instruction_end


def compute_llm_feedback(llm, qa_dataset, nx_graph_data, question_to_subgraphs_mapping):
    correct_answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    llm_feedbacks_dict = {}  # a dict of dicts in the form {node_type_0: {node_index_0: conf_diff_0, node_index_1: conf_diff_1...}, ...}

    for i, (qa_index, subgraph_tuples) in enumerate(question_to_subgraphs_mapping.items()):

        print(f'Processing question: {qa_index} ({i} / {len(question_to_subgraphs_mapping)})')
        # for node_index, question_node_representation in enumerate(batch['question']):
        # qa_index = batch['question'].node_uid[node_index].item()
        # subgraph_tuples = subgraphs_dict[qa_index]
        question_dict = dict(qa_dataset.iloc[qa_index].drop(['id', 'cop', 'exp']))
        correct_answer = qa_dataset.iloc[qa_index]['cop']
        prompt = """Question: {} A. {} B. {} C. {} D. {}""".format(
            question_dict['question'],
            question_dict['opa'],
            question_dict['opb'],
            question_dict['opc'],
            question_dict['opd']
        )

        # Process question without context
        llm.inference(prompt)
        llm_vanilla_response_dict = llm.get_confidence(correct_answer_map[correct_answer])

        if llm_vanilla_response_dict['confidence'] == -1:
            print(f'Wrong response format. Question {i} ignored')
            continue

        # Create LLMFeedback object
        llm_feedback = LLMFeedback(qa_index, llm_vanilla_response_dict['response'], llm_vanilla_response_dict['confidence'], llm_vanilla_response_dict['cop_confidence'], llm_vanilla_response_dict['accuracy'])

        # Process contexts
        for num_knowledge_nodes, (node_uid, node_type) in enumerate(subgraph_tuples):
            if num_knowledge_nodes >= 20:
                break
            # Create Context string
            node_name = nx_graph_data.nodes[node_uid]['name']
            context = f'Context: The {node_type} {node_name}. '
            prompt_with_context = context + prompt
            llm.inference(prompt_with_context)
            llm_context_response_dict = llm.get_confidence(correct_answer_map[correct_answer])

            if llm_context_response_dict['confidence'] == -1:
                print(f'Wrong response format. Question {i}, Node {node_uid} ignored')
                continue

            llm_feedback.insert_feedback(node_type, node_uid, llm_context_response_dict['confidence'], llm_context_response_dict['cop_confidence'], llm_context_response_dict['accuracy'])
            # confidence_diffs_dict[qa_index][node_type][node_uid] = question_confidence_diff

        llm_feedbacks_dict[qa_index] = llm_feedback
        llm_feedback.print_feedback()

        if i % 100 == 0:
            pickle.dump(llm_feedbacks_dict, open(os.path.join(ROOT_DIR, 'datasets', f'llm_feedbacks_{i}.pickle'), 'wb'))

    return llm_feedbacks_dict
