import os.path
import pickle
import random
import torch
import networkx as nx

from itertools import chain
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from typing import List, Tuple

from src.preprocess_graph.build_subgraph import convert_nx_to_hetero_data
from config import ROOT_DIR


class MedicalQADatasetBuilder:

    def __init__(self,
                 paths_to_dataset: List[str],
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 positive_relation_type: Tuple[str, str, str] = ('question', 'question_correct_answer', 'answer'),
                 neg_relation_type: Tuple[str, str, str] = ('question', 'question_wrong_answer', 'answer'),
                 disjoint_train_edges_ratio: float = 0.9,
                 negative_sampling_ratio: int = 3,
                 batch_size: int = 32):

        self.raw_data_list = self.build_raw_data_list(paths_to_dataset)
        self.all_edges_dict = {}
        self.processed_data_list = self.build_processed_data_list()

        self.num_val_samples = int(len(self.raw_data_list) * val_ratio)
        self.num_test_samples = int(len(self.raw_data_list) * test_ratio)
        self.num_train_samples = len(self.raw_data_list) - self.num_val_samples - self.num_test_samples

        self.positive_relation_type = positive_relation_type
        self.negative_relation_type = neg_relation_type
        self.disjoint_train_ratio = disjoint_train_edges_ratio
        self.negative_sampling_ratio = negative_sampling_ratio

        self.processed_train_dataset = self.processed_data_list[:self.num_train_samples].copy()
        self.processed_val_dataset = self.processed_data_list[: self.num_train_samples + self.num_val_samples].copy()
        self.processed_test_dataset = self.processed_data_list.copy()

        random.shuffle(self.processed_val_dataset)
        random.shuffle(self.processed_test_dataset)

        self.train_loader = DataLoader(self.processed_train_dataset, batch_size=batch_size)
        self.train_mini_batches = self.preprocess_batches(self.train_loader)
        self.train_edges_dict = self.find_edges_split(self.train_mini_batches)

        self.val_loader = DataLoader(self.processed_val_dataset, batch_size=batch_size)
        self.val_mini_batches = self.preprocess_batches(self.val_loader, is_train=False, edge_index_uids_dict=self.train_edges_dict)
        self.val_edge_dict = self.find_edges_split(self.val_mini_batches)

        self.test_loader = DataLoader(self.processed_test_dataset, batch_size=batch_size)
        self.test_mini_batches = self.preprocess_batches(self.test_loader, is_train=False, edge_index_uids_dict=self.val_edge_dict)

    def build_raw_data_list(self, root_dirs_list: List[str]):
        data_list = []

        for root_dir in root_dirs_list:

            file_names_list = os.listdir(os.path.join(ROOT_DIR, root_dir))

            for file_name in file_names_list:
                path = os.path.join(ROOT_DIR, root_dir, file_name)

                with open(path, 'rb') as f:
                    hetero_data = pickle.load(f)
                    data_list.append(hetero_data)

        return data_list

    def build_processed_data_list(self):

        processed_data_list = []
        edge_uid_offset = 0

        for graph in self.raw_data_list:
            hetero_data, edge_uid_offset = convert_nx_to_hetero_data(graph, edge_uid_offset=edge_uid_offset)
            processed_data_list.append(hetero_data)

            for edge_type in hetero_data.edge_types:
                if edge_type not in self.all_edges_dict:
                    self.all_edges_dict[edge_type] = [hetero_data[edge_type].edge_uid]
                else:
                    self.all_edges_dict[edge_type].append(hetero_data[edge_type].edge_uid)

        rev_types_dict = {}

        for edge_type, edge_uids_nested_list in self.all_edges_dict.items():
            self.all_edges_dict[edge_type] = torch.tensor(list(chain.from_iterable([x] if isinstance(x, int) else x for x in edge_uids_nested_list)))
            # rev_types_dict[(edge_type[2], f'rev_{edge_type[1]}', edge_type[0])] = self.all_edges_dict[edge_type]

        self.all_edges_dict.update(rev_types_dict)

        return processed_data_list

    def shuffle_dataset(self, dataset_list: List[HeteroData]):
        random.shuffle(dataset_list)
        return dataset_list

    def preprocess_batches(self, data_loader: DataLoader, is_train=True, edge_index_uids_dict=None):

        processed_batches = []

        for batch in data_loader:

            batch = self.ensure_batch_uniqueness(batch)
            if is_train:

                """
                The train dataset is the first one being processed --> no batch have been  used in other datasets.
                """

                num_positive_edges = batch[self.positive_relation_type].edge_index.size(1)

                positive_perm = torch.randperm(num_positive_edges)

                num_message_passing_edges = int(self.disjoint_train_ratio * num_positive_edges)

                positive_edge_index_indices = positive_perm[num_message_passing_edges:]
                positive_edge_label_index_indices = positive_perm[:num_message_passing_edges]

                # Find EdgeStore attributed for positive_relation_type
                positive_edge_index = batch[self.positive_relation_type].edge_index[:, positive_edge_index_indices]
                positive_edge_label_index = batch[self.positive_relation_type].edge_index[:, positive_edge_label_index_indices]
                positive_edge_label = torch.ones((1, positive_edge_label_index.size(1)))

            else:
                positive_edge_index_indices, positive_edge_label_index_indices = self.split_labels(batch, edge_index_uids_dict)

                if positive_edge_index_indices is None:
                    continue

                positive_edge_index = batch[self.positive_relation_type].edge_index[:, positive_edge_index_indices]
                positive_edge_label_index = batch[self.positive_relation_type].edge_index[:, positive_edge_label_index_indices]
                labels_length = 1 if positive_edge_label_index_indices.dim() == 0 else len(positive_edge_label_index_indices)
                positive_edge_label = torch.ones((1, labels_length))

            positive_edge_index_uids = batch[self.positive_relation_type].edge_uid[positive_edge_index_indices]
            positive_edge_label_uids = batch[self.positive_relation_type].edge_uid[positive_edge_label_index_indices]

            # Find the EdgeStore attributes for positive_relation_type (self.negative_ampler ensures each batch contain all answer possibilities per question)
            negative_edge_index, negative_edge_index_uids = self.negative_sampler(batch, positive_edge_index[0])
            negative_edge_label_index, negative_edge_label_uids = self.negative_sampler(batch, positive_edge_label_index[0])
            negative_edge_label = torch.zeros((1, negative_edge_label_index.size(1)))

            # Set EdgeStore attribute
            batch[self.positive_relation_type].edge_index = positive_edge_index
            batch[self.positive_relation_type].edge_label_index = positive_edge_label_index
            batch[self.positive_relation_type].edge_label = positive_edge_label
            batch[self.positive_relation_type].edge_index_uid = positive_edge_index_uids
            batch[self.positive_relation_type].edge_label_uid = positive_edge_label_uids

            batch[self.negative_relation_type].edge_index = negative_edge_index
            batch[self.negative_relation_type].edge_label_index = negative_edge_label_index
            batch[self.negative_relation_type].edge_label = negative_edge_label
            batch[self.negative_relation_type].edge_index_uid = negative_edge_index_uids
            batch[self.negative_relation_type].edge_label_uid = negative_edge_label_uids

            # Set EdgeStore attributes for the reverse relations
            rev_positive_relation_type = (self.positive_relation_type[2], f'rev_{self.positive_relation_type[1]}', self.positive_relation_type[0])
            rev_negative_relation_type = (self.negative_relation_type[2], f'rev_{self.negative_relation_type[1]}', self.negative_relation_type[0])

            batch[rev_positive_relation_type].edge_index = positive_edge_index.flip([0])
            batch[rev_positive_relation_type].edge_label_index = positive_edge_label_index.flip([0])
            batch[rev_positive_relation_type].edge_label = positive_edge_label
            batch[rev_positive_relation_type].edge_index_uid = positive_edge_index_uids
            batch[rev_positive_relation_type].edge_label_uid = positive_edge_label_uids

            batch[rev_negative_relation_type].edge_index = negative_edge_index.flip([0])
            batch[rev_negative_relation_type].edge_label_index = negative_edge_label_index.flip([0])
            batch[rev_negative_relation_type].edge_label = negative_edge_label
            batch[rev_negative_relation_type].edge_index_uid = negative_edge_index_uids
            batch[rev_negative_relation_type].edge_label_uid = negative_edge_label_uids

            processed_batches.append(batch)

        return processed_batches

    def negative_sampler(self, batch, source_node_indices):
        negative_examples = []
        negative_edge_uids = []
        negative_indices = batch[self.negative_relation_type].edge_index

        if source_node_indices.dim() == 0:
            source_node_indices = source_node_indices.unsqueeze(0)

        for index in source_node_indices:
            negative_example_indices = torch.where(negative_indices[0] == index)[0][:self.negative_sampling_ratio]
            negative_examples.append(negative_indices[:, negative_example_indices])
            negative_edge_uids.append(batch[self.negative_relation_type].edge_uid[negative_example_indices])

        return torch.cat(negative_examples, dim=1), torch.cat(negative_edge_uids)

    def split_labels(self, batch, edge_index_uids_dict):

        # Extract and edge_label

        edge_index_indices = []

        # Find all indices used as edge_index
        for value in edge_index_uids_dict[self.positive_relation_type]:
            index = torch.where(batch[self.positive_relation_type].edge_uid == int(value))[0]

            if index.numel() != 0:
                edge_index_indices.append(int(index))

        if len(edge_index_indices) == batch[self.positive_relation_type].edge_uid.size(-1):
            return None, None

        # Create a mask, separation edge_index indices from edge_label indices
        positive_edge_index_indices_mask = torch.ones(batch[self.positive_relation_type].edge_uid.size(-1), dtype=torch.bool)
        positive_edge_index_indices_mask[edge_index_indices] = False
        edge_label_index_indices = torch.nonzero(positive_edge_index_indices_mask).squeeze()

        return edge_index_indices, edge_label_index_indices

    def find_edges_split(self, batches_list):
        # Create edge_index_dict
        edge_index_uids_dict = {}

        for batch in batches_list:
            for edge_type in batch.edge_types:

                if edge_type not in edge_index_uids_dict:
                    edge_index_uids_dict[edge_type] = []

                edge_index_uids_dict[edge_type].append(batch[edge_type].edge_uid)

        for edge_type in edge_index_uids_dict.keys():
            edge_index_uids_dict[edge_type] = torch.cat(edge_index_uids_dict[edge_type], dim=0)

        return edge_index_uids_dict

    def ensure_batch_uniqueness(self, batch):
        for node_type in batch.node_types:
            edges_dict = {}

            for edge_type in batch.edge_types:

                if edge_type[0] == node_type:
                    edges_dict[edge_type] = 0
                elif edge_type[2] == node_type:
                    edges_dict[edge_type] = 1

            unique_node_type_features, unique_indices = torch.unique(batch[node_type].x, dim=0, return_inverse=True)

            if unique_node_type_features.size(1) == batch[node_type].x.size(1):
                continue

            batch[node_type].x = unique_node_type_features

            for edge_type, index in edges_dict.items():
                for j in range(batch[edge_type].edge_index.size(1)):
                    batch[edge_type].edge_index[index, j] = unique_indices[batch[edge_type].edge_index[index, j]]

        return batch


def build_merged_graphs_raw_dataset(graphs_list):
    merged_graph = nx.Graph()

    # Merge all graphs into the merged_graph
    for graph in graphs_list:
        merged_graph.add_nodes_from(graph.nodes(data=True))
        merged_graph.add_edges_from(graph.edges(data=True))

    return merged_graph
