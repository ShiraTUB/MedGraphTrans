import os
import torch
import pickle

from transformers import logging as hf_logging
from tqdm import tqdm

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
        self.input_encoding_size = 0

        # Set the LLM's logging level to ERROR to suppress lower level logs
        hf_logging.set_verbosity_error()

    def inference_batch(self, prompts):
        """
        Process a batch of prompts.
        """
        predictions, output_encodings = [], []
        batch_size = len(prompts)
        instruct_prompts = [self.format_prompt(prompt) for prompt in prompts]
        batch_encoding = self.tokenizer(instruct_prompts, padding=True, return_tensors="pt", truncation=True)
        batch_encoding = batch_encoding.to(self.device)
        self.input_encoding_size = batch_encoding.input_ids.shape[1]

        with torch.no_grad():
            model_outputs = self.model.generate(batch_encoding.input_ids, attention_mask=batch_encoding.attention_mask, output_scores=True, max_new_tokens=4, return_dict_in_generate=True)

        # Extract scores per prompt
        scores_per_prompt = [[] for _ in range(batch_size)]

        for token_scores in model_outputs.scores:
            # token_scores is of shape [batch_size, vocabulary_size]
            for i in range(batch_size):
                # Extract the scores for the i-th prompt
                prompt_scores = token_scores[i]
                scores_per_prompt[i].append(prompt_scores)

        scores = []
        for score in scores_per_prompt:
            for s in score:
                s = s.unsqueeze(0)
            scores.append(tuple(score))

        # Process and return the results
        for prompt_index in range(batch_size):
            # Extract the model's prediction indices for the current item
            response_ids = model_outputs.sequences[prompt_index].unsqueeze(0)
            response_scores = scores[prompt_index]
            output_encodings.append(response_ids)
            predictions.append(response_scores)

        return output_encodings, predictions

    def inference(self, prompt):

        instruct_prompt = self.format_prompt(prompt)

        input_encoding = self.tokenizer(instruct_prompt, return_tensors="pt")
        input_encoding = input_encoding.to(self.device)
        self.input_encoding_size = input_encoding.input_ids.shape[1]

        model_outputs = self.model.generate(input_encoding.input_ids,
                                            attention_mask=input_encoding.attention_mask,
                                            output_scores=True,
                                            max_new_tokens=4,
                                            return_dict_in_generate=True)

        predictions = model_outputs.scores
        output_encoding = model_outputs.sequences

        return output_encoding, predictions

    def get_confidence(self, correct_answer, output_encoding, predictions) -> dict:

        # Get the model's prediction indices
        response_ids = output_encoding[:, self.input_encoding_size:].squeeze(0).tolist()

        most_probable_tokens_ids = torch.topk(predictions[0], 100).indices.tolist()[0]

        # Get all tokens' probabilities
        all_tokens_probabilities = torch.nn.functional.softmax(predictions[0], dim=-1)

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

    def format_prompt(self, str):

        # str should have the strcture: Question: A headache could be a symptom of? A. Diabetes B. Concussion C. Broken leg D. Hemorrhoids

        instruction_beginning = "[INST] Your task is to answer multiple choice questions. You MUST respond only with a single letter of the correct answer. Question: Greenhouses are great for plants like A. Pizza B. Lollipops C. Candles D. French beans Answer: [/INST] D [INST] Your task is to answer multiple choice questions. You MUST respond only with a single letter of the correct answer. "
        instruction_end = " Answer: [/INST]"

        return instruction_beginning + str + instruction_end


def compute_llm_feedback(llm, qa_dataset, nx_graph_data, question_to_subgraphs_mapping):
    correct_answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    llm_feedbacks_dict = {}  # a dict of dicts in the form {node_type_0: {node_index_0: conf_diff_0, node_index_1: conf_diff_1...}, ...}

    for i, (qa_index, subgraph_tuples) in enumerate(tqdm(question_to_subgraphs_mapping.items())):

        qa_row = qa_dataset.iloc[qa_index]
        question_dict = dict(qa_row.drop(['id', 'cop', 'exp']))
        correct_answer = qa_row['cop']
        prompt = """Question: {} A. {} B. {} C. {} D. {}""".format(
            question_dict['question'],
            question_dict['opa'],
            question_dict['opb'],
            question_dict['opc'],
            question_dict['opd']
        )

        # Process question without context
        output_encodings, predictions = llm.inference(prompt)
        llm_vanilla_response_dict = llm.get_confidence(correct_answer_map[correct_answer], output_encodings, predictions)

        if llm_vanilla_response_dict['confidence'] == -1:
            print(f'Wrong response format. Question {i} ignored')
            continue

        # Create LLMFeedback object
        llm_feedback = LLMFeedback(qa_index, llm_vanilla_response_dict['response'], llm_vanilla_response_dict['confidence'], llm_vanilla_response_dict['cop_confidence'], llm_vanilla_response_dict['accuracy'])

        # Batch process contexts

        prompts_with_context = []

        for node_uid, node_type in subgraph_tuples[:20]:
            # Create Context string
            node_name = nx_graph_data.nodes[node_uid]['name']
            context = f'Context: The {node_type} {node_name}. '
            prompt_with_context = context + prompt
            prompts_with_context.append(prompt_with_context)

        # Batch inference
        batch_output_encodings, batch_predictions = llm.inference_batch(prompts_with_context)

        # Postprocess batch model output
        for j, (node_uid, node_type) in enumerate(subgraph_tuples[:20]):
            output_encoding = batch_output_encodings[j]
            prediction = batch_predictions[j]
            llm_context_response_dict = llm.get_confidence(correct_answer_map[correct_answer], output_encoding=output_encoding, predictions=prediction)

            if llm_context_response_dict['confidence'] == -1:
                print(f'Wrong response format. Node {node_uid} ignored')
                continue

            llm_feedback.insert_feedback(node_type, node_uid, llm_context_response_dict['confidence'], llm_context_response_dict['cop_confidence'], llm_context_response_dict['accuracy'])

            llm_feedbacks_dict[qa_index] = llm_feedback

        if i % 100 == 0:
            print(f'Example Feedback:\n')
            llm_feedbacks_dict[qa_index].print_feedback()
            pickle.dump(llm_feedbacks_dict, open(os.path.join(ROOT_DIR, 'datasets', f'llm_feedbacks_{i}.pickle'), 'wb'))

    return llm_feedbacks_dict
