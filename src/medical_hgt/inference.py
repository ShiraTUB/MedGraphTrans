import torch
import pickle
import os
import argparse

from src.medical_hgt.model import MedicalHGT
from src.medical_hgt.ml_utils import find_most_relevant_nodes, query_chatbot

from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Inference HGT on PrimeKG + Medmcqa')
parser.add_argument('--model_path', type=str, default='experiments/linkneighbor-3.0-4,3,2,10,10,3-128_run1.pth', help='Path of target model to load')
parser.add_argument('--hetero_data_path', type=str, default='datasets/merged_hetero_dataset/processed_graph_480_test_masked_with_edge_uids.pickle', help='Path of the test dataset')

args = parser.parse_args()


def load_model():
    model = MedicalHGT(hidden_channels=64)

    with open(os.path.join(ROOT_DIR, args.model_path), 'rb') as f:
        saved_object = torch.load(f)
        model.load_state_dict(saved_object.state_dict)

    model.eval()

    return model


# def add_and_predict_link(hetero_data, model):
#
#     # Predict links
#     with torch.no_grad():
#         pred = model(hetero_data)
#
#         eval_pred = pred.detach()
#         eval_y = hetero_data["question", "question_answer", "answer"].edge_label.detach()

def find_context(question_graph_data, model, prime_kg):
    with torch.no_grad():
        _, _, nodes_representation_dict = model(question_graph_data)
    knowledge_nodes_uids_dict = {}
    for node_type in question_graph_data.node_types:
        if node_type != 'question' and node_type != 'answer':
            knowledge_nodes_uids_dict[node_type] = question_graph_data[node_type].node_uid

    question_node_representation = nodes_representation_dict['question']
    relevant_knowledge_nodes = find_most_relevant_nodes(question_graph_data, nodes_representation_dict, question_node_representation, knowledge_nodes_uids_dict, prime_kg)
    return relevant_knowledge_nodes


def test_model(qa_dataset, hetero_graph_dataset, prime_kg):

    # todo WIP

    model = load_model()
    correct_answer_dict = {0: 'opa', 1: 'opb', 2: 'opc', 3: 'opd'}
    output_instructions = f'Return only the correct answer as a single letter a-d.'
    output_without_context_list = []
    output_with_context_list = []

    for index, row in qa_dataset:
        question_graph = hetero_graph_dataset[index]
        context = find_context(question_graph, model, prime_kg)
        dataset_question_dict = dict(row)

        correct_answer = row['cop']

        response_without_context = query_chatbot(str(dataset_question_dict), output_instructions)
        dataset_question_dict['context'] = context
        response_with_context = query_chatbot(str(dataset_question_dict), output_instructions)

        output_without_context_list.append(response_without_context)
        output_with_context_list.append(response_with_context)

    accuracy_without_context = compute_chat_accuracy(output_without_context_list)
    accuracy_with_context = compute_chat_accuracy(output_with_context_list)


with open(os.path.join(ROOT_DIR, args.hetero_data_path), 'rb') as f:
    hetero_data = pickle.load(f)
