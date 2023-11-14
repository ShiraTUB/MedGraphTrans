import torch
import numpy as np
from torch_geometric.data import HeteroData


def split_data(hetero_data, labels_ratio=0.3, neg_labels_ratio=1):
    train_data = HeteroData()
    val_data = HeteroData()
    test_data = HeteroData()

    for node_type, node_features in hetero_data.x_dict.items():
        train_data[node_type].x = node_features
        val_data[node_type].x = node_features
        test_data[node_type].x = node_features

    for edge_type, edge_indices in hetero_data.edge_index_dict.items():

        if hasattr(hetero_data[edge_type], "train_mask"):
            train_mask = hetero_data[edge_type].train_mask
            val_mask = hetero_data[edge_type].val_mask
            test_mask = hetero_data[edge_type].test_mask

            train_indices_all = train_mask.nonzero().squeeze()
            val_indices_all = val_mask.nonzero().squeeze()
            test_indices_all = test_mask.nonzero().squeeze()

            val_label_len = int(len(val_indices_all) * labels_ratio)
            test_label_len = int(len(test_indices_all) * labels_ratio)

            val_label_indices = val_indices_all[:val_label_len]
            val_indices = val_indices_all[val_label_len:]

            test_label_indices = test_indices_all[:test_label_len]
            test_indices = test_indices_all[test_label_len:]

            train_data[edge_type].edge_index = (edge_indices.T[train_indices_all]).T
            train_data[edge_type].edge_label_index = (edge_indices.T[train_indices_all]).T
            train_data[edge_type].edge_label = torch.ones(train_data[edge_type].edge_index.size(1))

            val_data[edge_type].edge_index = (edge_indices.T[val_indices]).T
            pos_val_label_indices = (edge_indices.T[val_label_indices]).T
            neg_val_label_indices = negative_sampling(len(val_label_indices) * neg_labels_ratio, (edge_indices.T[val_indices_all]).T)
            val_data[edge_type].edge_label_index = torch.cat((pos_val_label_indices, neg_val_label_indices), dim=1)
            val_data[edge_type].edge_label = torch.ones(pos_val_label_indices.size(1) + neg_val_label_indices.size(1))
            val_data[edge_type].edge_label[pos_val_label_indices.size(1):] = 0

            test_data[edge_type].edge_index = (edge_indices.T[test_indices]).T
            pos_test_label_indices = (edge_indices.T[test_label_indices]).T
            neg_test_label_indices = negative_sampling(len(test_label_indices) * neg_labels_ratio, (edge_indices.T[test_indices_all]).T)
            test_data[edge_type].edge_label_index = torch.cat((pos_test_label_indices, neg_test_label_indices), dim=1)
            test_data[edge_type].edge_label = torch.ones(pos_test_label_indices.size(1) + neg_test_label_indices.size(1))
            test_data[edge_type].edge_label[pos_test_label_indices.size(1):] = 1

        else:
            train_data[edge_type].edge_index = edge_indices
            val_data[edge_type].edge_index = edge_indices
            test_data[edge_type].edge_index = edge_indices

    return train_data, val_data, test_data


def negative_sampling(num_samples, edge_indices):
    # Determine the max values for each row
    max_values = edge_indices.max(axis=1)

    # Create a set of tuples from A for easy comparison
    existing_pairs = set(tuple(pair) for pair in edge_indices.T)

    # Initialize list for new pairs
    negative_edges = []

    random_source_indices = np.random.permutation(np.arange(max_values[0][0] + 1))
    random_target_indices = np.random.permutation(np.arange(max_values[0][1] + 1))

    i = 0
    j = 0

    # Generate new pairs
    while len(negative_edges) < num_samples:
        if (random_source_indices[i], random_target_indices[j]) not in existing_pairs:
            negative_edges.append([random_source_indices[i], random_target_indices[j]])
            i += 1
            j += 1

    # Convert new pairs to a numpy aray
    negative_samples = torch.tensor(negative_edges).T

    return negative_samples
