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
from src.medical_hgt.train import train_model
from src.medical_hgt.dataset_builder import MedicalQADatasetBuilder
from src.medical_hgt.llm import LLM, LLMFeedback
from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')
parser.add_argument('--dataset_path', type=str, default='datasets/merged_hetero_dataset/processed_graph_1000_train_val_masked_with_edge_uids_and_neg_edges.pickle', help='Path of the processed dataset')
parser.add_argument('--dataset_folders_paths', type=List[str], default=['datasets/graph_dataset_30_11_23/train'], help='Paths to the folders containing raw graphs datasets')
parser.add_argument('--experiment_output_path', type=str, default='experiments', help='Path of the target experiments folder')
parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--train_dataset', type=str, default='datasets/train_data_01_12_23.pickle', help='Processed train data pickle path')
parser.add_argument('--val_dataset', type=str, default='datasets/val_data_01_12_23.pickle', help='Processed val data pickle path')
parser.add_argument('--test_dataset', type=str, default='datasets/test_data_01_12_23.pickle', help='Processed test data pickle path')
parser.add_argument('--llm_feedback', type=str, default='datasets/llm_feedback/llm_feedbacks_test_100.pickle', help='Processed llm feedback')
parser.add_argument('--kg_subgraphs_mapping', type=str, default='datasets/subgraphs_dict.pickle', help='Mapping between question ids and their respective subgraphs')

args = parser.parse_args()


def run_experiment(data_loader_params, device, llm, llm_feedbacks_dict, question_to_subgraphs_mapping, prime_kg, qa_dataset, runs=1, loaders=None):
    """Runs a multi-trial experiment using the given DataLoaderParams."""
    # todo: instead of calling build_link_neighbor_loaders call MedicalQADatasetBuilder (make necessary adjustments) - loaders = dataset_builder.train_mini_batches, dataset_builder.val_mini_batches, dataset_builder.test_mini_batched
    if loaders is None:
        dataset_builder = MedicalQADatasetBuilder(
            args.dataset_folders_paths,
            val_ratio=data_loader_params.val_ratio,
            test_ratio=data_loader_params.test_ratio,
            disjoint_train_edges_ratio=data_loader_params.disjoint_train_edges_ratio,
            negative_sampling_ratio=data_loader_params.negative_sampling_ratio,
            batch_size=data_loader_params.batch_size)

        loaders = {'train': dataset_builder.train_mini_batches, 'val': dataset_builder.val_mini_batches, 'test': dataset_builder.test_mini_batches}

    for i in range(runs):
        file_name = data_loader_params.get_file_name() + f'_run{i + 1}.pth'
        medical_hgt = MedicalHGT(hidden_channels=64, out_channels=64, num_heads=4, num_layers=1)
        train_model(medical_hgt=medical_hgt,
                    split_loaders=loaders,
                    device=device,
                    file_name=file_name,
                    num_epochs=data_loader_params.num_epochs,
                    llm_feedbacks_dict=llm_feedbacks_dict,
                    prime_kg=prime_kg,
                    qa_dataset=qa_dataset,
                    question_to_subgraphs_mapping=question_to_subgraphs_mapping,
                    llm=llm)


@dataclass(frozen=True)
class DataLoaderParams:
    """Helper class for holding the parameters of LinkNeighborLoader."""
    val_ratio: float
    test_ratio: float
    disjoint_train_edges_ratio: float
    negative_sampling_ratio: int
    batch_size: int
    num_epochs: int

    def get_file_name(self):
        """Generates the file name for storing the results, based on the params."""
        folder_path = os.path.join(ROOT_DIR, args.experiment_output_path)
        return f'{folder_path}/dataloader-{self.negative_sampling_ratio}-{self.val_ratio}-{self.test_ratio}-{self.negative_sampling_ratio}-{self.batch_size}'


data_loader_params_list = [
    # baseline
    DataLoaderParams(val_ratio=0.1, test_ratio=0.1, disjoint_train_edges_ratio=0.9, negative_sampling_ratio=3, batch_size=16, num_epochs=100),
    # different batch sizes
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=512),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=256),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=64),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=32),
    # # different sampling ratios
    # DataLoaderParams(neg_sampling_ratio=8.0, num_neighbors=[20, 10], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=4.0, num_neighbors=[20, 10], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=1.0, num_neighbors=[20, 10], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=0.5, num_neighbors=[20, 10], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=0.1, num_neighbors=[20, 10], batch_size=128),
    # # single-hop
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[2], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[4], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[10], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[40], batch_size=128),
    # # 2-hop with different sizes
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[2, 1], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[10, 5], batch_size=128),
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[40, 20], batch_size=128),
    # # 3-hop
    # DataLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10, 5], batch_size=128),
]

# Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_path = os.path.join(ROOT_DIR, 'datasets/train_data_batch_size_16.pickle')
val_data_path = os.path.join(ROOT_DIR, 'datasets/val_data_batch_size_16.pickle')
test_data_path = os.path.join(ROOT_DIR, 'datasets/test_data_batch_size_16.pickle')

train_data = pickle.load(open(train_data_path, 'rb'))
val_data = pickle.load(open(val_data_path, 'rb'))
test_data = pickle.load(open(test_data_path, 'rb'))

loaders = {'train': train_data, 'val': val_data, 'test': test_data}

prime_kg = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/primeKG_nx_medium.pickle'), 'rb'))

llm_feedbacks_dict = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/llm_feedback/llm_feedbacks_test_100.pickle'), 'rb'))
question_to_subgraphs_mapping = pickle.load(open(os.path.join(ROOT_DIR, 'datasets/subgraphs_dict.pickle'), 'rb'))

qa_dataset = load_dataset("medmcqa")
qa_dataset = pd.DataFrame(qa_dataset['train'])

# Load LLM

# 4bit quantization config

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = None  # LLM(model, tokenizer)
for data_loader_params in data_loader_params_list:
    run_experiment(data_loader_params, device, llm, llm_feedbacks_dict, question_to_subgraphs_mapping, prime_kg, qa_dataset)
