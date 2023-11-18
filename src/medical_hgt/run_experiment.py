import os
import pickle
import torch
import argparse

from dataclasses import dataclass
from typing import List

from src.medical_hgt.model import Model
from src.medical_hgt.training import train_model
from src.medical_hgt.data_loaders import build_link_neighbor_loaders
from src.medical_hgt.dataset_builder import MedicalQADatasetBuilder
from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')
parser.add_argument('--dataset_path', type=str, default='datasets/merged_hetero_dataset/processed_graph_1000_train_val_masked_with_edge_uids_and_neg_edges.pickle', help='Path of the processed dataset')
parser.add_argument('--dataset_folders_paths', type=List[str], default=['datasets/raw_graph_dataset_with_negative_edges/train', 'datasets/raw_graph_dataset_with_negative_edges/validation'],
                    help='Paths to the folders containing raw graphs datasets')
parser.add_argument('--experiment_output_path', type=str, default='experiments', help='Path of the target experiments folder')

args = parser.parse_args()


def run_experiment(link_neighbor_params, device, runs=2):
    """Runs a multi-trial experiment using the given LinkNeighborLoaderParams."""
    # todo: instead of calling build_link_neighbor_loaders call MedicalQADatasetBuilder (make necessary adjustments) - loaders = dataset_builder.train_mini_batches, dataset_builder.val_mini_batches, dataset_builder.test_mini_batched
    dataset_builder = MedicalQADatasetBuilder(args.dataset_folders_paths, batch_size=64)

    # loaders = build_link_neighbor_loaders(data,
    #                                       link_neighbor_params.neg_sampling_ratio,
    #                                       link_neighbor_params.num_neighbors,
    #                                       link_neighbor_params.batch_size)

    loaders = {'train': dataset_builder.train_mini_batches, 'val': dataset_builder.val_mini_batches, 'test': dataset_builder.test_mini_batches}

    all_edges_dict = dataset_builder.all_edges_dict

    for i in range(runs):
        file_name = link_neighbor_params.get_file_name() + f'_run{i + 1}.pth'
        model = Model(all_edges_dict, hidden_channels=64)
        train_model(model, loaders, device, file_name, num_epochs=link_neighbor_params.num_epochs)


@dataclass(frozen=True)
class LinkNeighborLoaderParams:
    """Helper class for holding the parameters of LinkNeighborLoader."""
    neg_sampling_ratio: float
    num_neighbors: list
    batch_size: int
    num_epochs: int
    directed: bool = False

    def get_file_name(self):
        """Generates the file name for storing the results, based on the params."""
        num_str = ','.join([str(n) for n in self.num_neighbors])
        dir_str = '-dir' if self.directed else ''
        folder_path = os.path.join(ROOT_DIR, args.experiment_output_path)
        return f'{folder_path}/linkneighbor-{self.neg_sampling_ratio}-{num_str}-{self.batch_size}{dir_str}'


link_neighbor_params_list = [
    # baseline
    LinkNeighborLoaderParams(neg_sampling_ratio=3.0, num_neighbors=[-1], batch_size=64, num_epochs=8),
    # different batch sizes
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=512),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=256),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=64),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10], batch_size=32),
    # # different sampling ratios
    # LinkNeighborLoaderParams(neg_sampling_ratio=8.0, num_neighbors=[20, 10], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=4.0, num_neighbors=[20, 10], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=1.0, num_neighbors=[20, 10], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=0.5, num_neighbors=[20, 10], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=0.1, num_neighbors=[20, 10], batch_size=128),
    # # single-hop
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[2], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[4], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[10], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[40], batch_size=128),
    # # 2-hop with different sizes
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[2, 1], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[10, 5], batch_size=128),
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[40, 20], batch_size=128),
    # # 3-hop
    # LinkNeighborLoaderParams(neg_sampling_ratio=2.0, num_neighbors=[20, 10, 5], batch_size=128),
]

# with open(os.path.join(ROOT_DIR, args.dataset_path), 'rb') as f:
#     hetero_data = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for link_neighbor_params in link_neighbor_params_list:
    run_experiment(link_neighbor_params, device)
