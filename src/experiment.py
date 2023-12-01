import torch
import pickle
import os
import argparse
import pandas as pd

from datasets import load_dataset
from src.medical_hgt.ml_utils import query_chatbot
from src.medical_hgt.inference import test_model

from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Run the MedGraphTrans experiment')
parser.add_argument('--model_path', type=str, default='experiments/linkneighbor-3.0-4,3,2,10,10,3-128_run1.pth', help='Path of target model to load')
parser.add_argument('--hetero_data_path', type=str, default='datasets/merged_hetero_dataset/processed_graph_480_test_masked_with_edge_uids.pickle', help='Path of the test dataset')
parser.add_argument('--qa_dataset_name', type=str, default='medmcqa', help='Name of dataset to download using datasets')
parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--save_results_path', type=str, default='datasets/primeKG_nx_medium.pickle', help='Target directory path to save the results')
parser.add_argument('--dataset_with_results_path', type=str, default=None, help='Path to dataset with chatbot results')

args = parser.parse_args()


def answer_qa_no_context(qa_dataset_df):
    #  prompt Chatbot with the questions and log the results
    dataset_with_lm_result = qa_dataset_df.copy()
    dataset_with_lm_result["response"] = None

    output_instructions = f'Return only the correct answer as a single letter a-d.'

    for index, row in qa_dataset_df.iterrows():
        dataset_question_dict = row.drop(['id', 'cop', 'exp']).to_dict()
        chatbot_response = query_chatbot(str(dataset_question_dict), output_instructions)
        dataset_with_lm_result.at[index, 'response'] = chatbot_response

    # save results
    dataset_with_lm_result.to_json(os.path.join(ROOT_DIR, f'{args.save_results_path}/dataset_with_lm_result.json'), orient='records')
    return dataset_with_lm_result


# load data
with open(os.path.join(ROOT_DIR, args.hetero_data_path), 'rb') as f:
    hetero_data = pickle.load(f)

with open(os.path.join(ROOT_DIR, args.prime_kg_dataset), 'rb') as f:
    kg = pickle.load(f)

qa_dataset = load_dataset(os.path.join(ROOT_DIR, args.qa_dataset_name))['validation']
qa_dataset_dataframe = pd.DataFrame(qa_dataset)

# Generate original dataset results
if args.dataset_with_results_path is None:
    dataset_with_lm_result = answer_qa_no_context(qa_dataset_dataframe)
else:
    dataset_with_lm_result = pd.read_json(os.path.join(ROOT_DIR, args.dataset_with_results_path))

# Generate enriched dataset results
chatbot_results_with_context_list = test_model(qa_dataset_dataframe, hetero_data, kg)

dataset_with_lm_result['response_with_context'] = chatbot_results_with_context_list

# save the final processed df -> with the chatbot's responses with and without context
dataset_with_lm_result.to_json(os.path.join(ROOT_DIR, f'{args.save_results_path}/dataset_with_lm_result_with_context.json'), orient='records')
