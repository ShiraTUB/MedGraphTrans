import os
import pickle
import torch
import argparse

import pandas as pd
import transformers

from datasets import load_dataset
from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.medical_hgt.model import MedicalHGT
from src.medical_hgt.train import train
from src.medical_hgt.dataset_builder import MedicalQADatasetBuilder, MedMCQADatasetBuilder
from src.medical_hgt.llm import LLM, LLMFeedback
from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')
parser.add_argument('--experiment_output_path', type=str, default='experiments', help='Path of the target experiments folder')
parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--train_dataset', type=str, default='datasets/train_data_01_12_23.pickle', help='Processed train data pickle path')
parser.add_argument('--val_dataset', type=str, default='datasets/val_data_01_12_23.pickle', help='Processed val data pickle path')
parser.add_argument('--test_dataset', type=str, default='datasets/test_data_01_12_23.pickle', help='Processed test data pickle path')
parser.add_argument('--llm_feedback', type=str, default='datasets/llm_feedback/llm_feedbacks_test_100.pickle', help='Processed llm feedback')
parser.add_argument('--kg_subgraphs_mapping', type=str, default='datasets/subgraphs_dict.pickle', help='Mapping between question ids and their respective subgraphs')

args = parser.parse_args()


@dataclass(frozen=True)
class ExperimentsParams:
    """Helper class for holding the parameters of a training experiment."""
    num_epochs: int
    lr: float
    channels: int
    num_heads: int
    num_layers: int

    def get_file_name(self):
        """Generates the file name for storing the results, based on the params."""
        folder_path = os.path.join(ROOT_DIR, 'experiments')
        return f'{folder_path}/experiment-{self.num_epochs}_epochs-{self.lr}_lr-{self.channels}_channels-{self.num_heads}_head-{self.num_layers}_layers'


def run_experiments(experiment_params, device, llm, train_llm_feedbacks_dict, val_llm_feedbacks_dict, question_to_subgraphs_mapping, prime_kg, qa_dataset, data_loaders):
    """Runs a multi-trial experiment using the given ExperimentsParams."""

    file_name = f'{experiment_params.get_file_name()}.pth'
    medical_hgt = MedicalHGT(channels=experiment_params.channels, num_heads=experiment_params.num_heads, num_layers=experiment_params.num_layers)
    medical_hgt_result = train(llm=llm,
                               medical_hgt=medical_hgt,
                               split_loaders=data_loaders,
                               device=device,
                               file_name=file_name,
                               qa_dataset=qa_dataset,
                               prime_kg=prime_kg,
                               train_llm_feedbacks_dict=train_llm_feedbacks_dict,
                               val_llm_feedbacks_dict=val_llm_feedbacks_dict,
                               question_to_subgraphs_mapping=question_to_subgraphs_mapping,
                               num_epochs=experiment_params.num_epochs,
                               lr=experiment_params.lr)

    return medical_hgt_result


experiments_list = [
    # baseline
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=4, num_layers=2),

    # learning rate
    ExperimentsParams(num_epochs=10, lr=0.0001, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.00025, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.0005, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.00075, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.002, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.003, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.004, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.005, channels=64, num_heads=4, num_layers=2),

    # hidden and out channels
    ExperimentsParams(num_epochs=10, lr=0.001, channels=128, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.001, channels=32, num_heads=4, num_layers=2),

    # num_heads and num_layers
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=2, num_layers=1),
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=2, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=2, num_layers=3),
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=4, num_layers=1),
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=4, num_layers=2),
    ExperimentsParams(num_epochs=10, lr=0.001, channels=64, num_heads=4, num_layers=3)

]

# Load data

train_data_path = os.path.join(ROOT_DIR, 'datasets/train_data_batch_size_16.pickle')
val_data_path = os.path.join(ROOT_DIR, 'datasets/val_data_batch_size_16.pickle')
test_data_path = os.path.join(ROOT_DIR, 'datasets/test_data_batch_size_16.pickle')

train_data = pickle.load(open(os.path.join(ROOT_DIR, args.train_dataset), 'rb'))
val_data = pickle.load(open(os.path.join(ROOT_DIR, args.val_dataset), 'rb'))
test_data = pickle.load(open(os.path.join(ROOT_DIR, args.test_dataset), 'rb'))

loaders = {'train': train_data, 'val': val_data, 'test': test_data}

llm_feedbacks_dict1 = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/llm_feedbacks_17839.pickle'), 'rb'))
llm_feedbacks_dict2 = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/llm_feedbacks_17898.pickle'), 'rb'))

prime_kg = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/prime_kg_nx_63960.pickle'), 'rb'))

question_to_subgraphs_mapping = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/question_to_subgraphs_mapping_17898.pickle'), 'rb'))

qa_dataset = load_dataset("medmcqa")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load LLM

# 4bit quantization config

# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = None  # LLM(model, tokenizer)
for experiment_params in experiments_list:
    experiment_results_list = []
    experiment_result = run_experiments(experiment_params,
                                        device=device,
                                        llm=llm,
                                        train_llm_feedbacks_dict=llm_feedbacks_dict2,
                                        val_llm_feedbacks_dict=llm_feedbacks_dict1,
                                        question_to_subgraphs_mapping=question_to_subgraphs_mapping,
                                        prime_kg=prime_kg,
                                        qa_dataset=qa_dataset,
                                        data_loaders=loaders)

    experiment_results_list.append(experiment_result)
