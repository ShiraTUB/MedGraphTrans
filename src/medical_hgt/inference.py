import torch
import pickle
import os
import argparse
import pandas as pd

from datasets import load_dataset
from src.medical_hgt.model import MedicalHGT
from src.medical_hgt.ml_utils import find_most_relevant_nodes, query_chatbot

from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Inference HGT on PrimeKG + Medmcqa')
parser.add_argument('--model_path', type=str, default='experiments/linkneighbor-3.0-4,3,2,10,10,3-128_run1.pth', help='Path of target model to load')
parser.add_argument('--hetero_data_path', type=str, default='datasets/merged_hetero_dataset/processed_graph_480_test_masked_with_edge_uids.pickle', help='Path of the test dataset')
parser.add_argument('--qa_dataset_name', type=str, default='medmcqa', help='Name of dataset to download using datasets')
parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--results_path', type=str, default='datasets/primeKG_nx_medium.pickle', help='Target directory path to save the results')

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


def test_model(qa_dataset_df, hetero_graph_dataset, prime_kg):
    # todo WIP

    #  prompt Chatbot with the context enriched questions and log the results
    chatbot_responses_list = []

    model = load_model()

    output_instructions = f'Return only the correct answer as a single letter a-d.'

    for index, row in qa_dataset_df.iterrows():
        question_graph = hetero_graph_dataset[index]
        context = find_context(question_graph, model, prime_kg)
        dataset_question_dict = row.drop(['id', 'cop', 'exp']).to_dict()
        dataset_question_dict['context'] = context
        chatbot_response = query_chatbot(str(dataset_question_dict), output_instructions)
        chatbot_responses_list.append(chatbot_response)

    return chatbot_responses_list


# load data
with open(os.path.join(ROOT_DIR, args.hetero_data_path), 'rb') as f:
    hetero_data = pickle.load(f)

with open(os.path.join(ROOT_DIR, args.hetero_data_path), 'rb') as f:
    kg = pickle.load(f)

qa_dataset = load_dataset(args.qa_dataset_name)['validation']
qa_dataset_dataframe = pd.DataFrame(qa_dataset)

test_model(qa_dataset_dataframe, hetero_data, kg)


