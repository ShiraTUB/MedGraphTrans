import torch
from torch_geometric.data import HeteroData


def split_data(hetero_data: HeteroData, positive_target_relation: tuple[str, str, str] = ('question', 'question_correct_answer', 'answer'), negative_target_relation: tuple[str, str, str] = ('question', 'question_wrong_answer', 'answer'),
               negative_sampling_ration=3, add_negative_train_samples: bool = False):
    """
    Args:
        hetero_data: the dataset to split
        positive_target_relation: the relation type used for positive examples sampling. It is expected that edge of this type will have the attributes train_mask, val_mask and test_mask.
        negative_target_relation: the relation type used for negative examples sampling
        negative_sampling_ration: the ration of negative to positive examples
        add_negative_train_samples: if set to True, negative examples will be added to the train_data as well

    Returns:
        (train_data: HeteroData, val_Data: HeteroData, test_data: HeteroData)
    """

    train_data = HeteroData()
    val_data = HeteroData()
    test_data = HeteroData()

    # All nodes have the same features across all datasets
    for node_type, node_features in hetero_data.x_dict.items():
        train_data[node_type].x = node_features
        train_data[node_type].node_uid = hetero_data[node_type].node_uid
        val_data[node_type].x = node_features
        val_data[node_type].node_uid = hetero_data[node_type].node_uid
        test_data[node_type].x = node_features
        test_data[node_type].node_uid = hetero_data[node_type].node_uid

    for edge_type, edge_indices in hetero_data.edge_index_dict.items():

        if edge_type == positive_target_relation:

            # Extract train, val and test masks and their respective indices
            train_mask = hetero_data[edge_type].train_mask
            val_mask = hetero_data[edge_type].val_mask
            test_mask = hetero_data[edge_type].test_mask

            train_indices = train_mask.nonzero().squeeze()
            val_indices = val_mask.nonzero().squeeze()
            test_indices = test_mask.nonzero().squeeze()

            # Set train_data attributes:
            pos_train_indices = edge_indices.T[train_indices].T
            train_data[edge_type].edge_index = pos_train_indices
            train_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid[train_indices]

            if add_negative_train_samples:
                neg_train_label_indices = negative_sampler(pos_train_indices, hetero_data[negative_target_relation].edge_index, negative_sampling_ration)
                train_edge_label_index = torch.cat((pos_train_indices, neg_train_label_indices), dim=1)
                train_edge_label = torch.ones(pos_train_indices.size(1) + neg_train_label_indices.size(1))
                train_edge_label[pos_train_indices.size(1):] = 0

                # shuffle labels
                indices_permutation = torch.randperm(train_edge_label_index.size(1))
                shuffled_label_indices = train_edge_label_index.T[indices_permutation].T
                shuffled_labels = train_edge_label[indices_permutation]
                train_data[edge_type].edge_label_index = shuffled_label_indices
                train_data[edge_type].edge_label = shuffled_labels

            else:
                train_data[edge_type].edge_label_index = pos_train_indices
                train_data[edge_type].edge_label = torch.ones(pos_train_indices.size(1))

            # Set val_data attributes
            pos_val_indices = edge_indices.T[val_indices].T
            neg_val_label_indices = negative_sampler(pos_val_indices, hetero_data[negative_target_relation].edge_index, negative_sampling_ration)
            val_edge_label_index = torch.cat((pos_val_indices, neg_val_label_indices), dim=1)
            val_edge_label = torch.ones(pos_val_indices.size(1) + neg_val_label_indices.size(1))
            val_edge_label[pos_val_indices.size(1):] = 0

            # shuffle labels
            indices_permutation = torch.randperm(val_edge_label_index.size(1))
            shuffled_label_indices = val_edge_label_index.T[indices_permutation].T
            shuffled_labels = val_edge_label[indices_permutation]

            val_data[edge_type].edge_index = pos_val_indices
            val_data[edge_type].edge_label_index = shuffled_label_indices
            val_data[edge_type].edge_label = shuffled_labels
            val_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid[val_indices]

            # Set test_data attributes
            pos_test_indices = edge_indices.T[test_indices].T
            neg_test_label_indices = negative_sampler(pos_val_indices, hetero_data[negative_target_relation].edge_index, negative_sampling_ration)
            test_edge_label_index = torch.cat((pos_test_indices, neg_test_label_indices), dim=1)
            test_edge_label = torch.ones(pos_test_indices.size(1) + neg_test_label_indices.size(1))
            test_edge_label[pos_test_indices.size(1):] = 0

            # shuffle labels
            indices_permutation = torch.randperm(test_edge_label_index.size(1))
            shuffled_label_indices = test_edge_label_index.T[indices_permutation].T
            shuffled_labels = test_edge_label[indices_permutation]

            test_data[edge_type].edge_index = pos_test_indices
            test_data[edge_type].edge_label_index = shuffled_label_indices
            test_data[edge_type].edge_label = shuffled_labels
            test_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid[test_indices]

        elif edge_type[1] == f'rev_{positive_target_relation[1]}':
            train_data[edge_type].edge_index = train_data[positive_target_relation].edge_index[[1, 0]]
            val_data[edge_type].edge_index = val_data[positive_target_relation].edge_index[[1, 0]]
            test_data[edge_type].edge_index = test_data[positive_target_relation].edge_index[[1, 0]]
            train_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid[train_indices]
            val_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid[val_indices]
            test_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid[test_indices]

            train_data[edge_type].edge_label_index = train_data[positive_target_relation].edge_label_index[[1, 0]]
            val_data[edge_type].edge_label_index = val_data[positive_target_relation].edge_label_index[[1, 0]]
            test_data[edge_type].edge_label_index = test_data[positive_target_relation].edge_label_index[[1, 0]]

            train_data[edge_type].edge_label = train_data[positive_target_relation].edge_label
            val_data[edge_type].edge_label = val_data[positive_target_relation].edge_label
            test_data[edge_type].edge_label = test_data[positive_target_relation].edge_label

        elif edge_type[1] == negative_target_relation[1] or edge_type[1] == f'rev_{negative_target_relation[1]}':
            continue

        else:
            train_data[edge_type].edge_index = edge_indices
            val_data[edge_type].edge_index = edge_indices
            test_data[edge_type].edge_index = edge_indices
            train_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid
            val_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid
            test_data[edge_type].edge_uid = hetero_data[edge_type].edge_uid

    return train_data, val_data, test_data


def negative_sampler(source_node_indices, negative_indices: torch.Tensor, negative_sampling_ration=3):
    negative_examples = []
    for index in source_node_indices[0]:
        negative_example_indices = torch.where(negative_indices[0] == index)[0][:negative_sampling_ration]
        negative_examples.append(negative_indices.T[negative_example_indices].T)

    return torch.cat(negative_examples, dim=1)
