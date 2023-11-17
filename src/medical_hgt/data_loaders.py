import torch_geometric.transforms as T

from torch_geometric.loader import LinkNeighborLoader
from src.medical_hgt.ml_utils import split_data


def random_link_split(data, val_neg_sampling_ratio):
    """Splits the data into train, test, and val data."""
    # transform = T.RandomLinkSplit(
    #     num_val=0.1,
    #     num_test=0.1,
    #     is_undirected=True,
    #     disjoint_train_ratio=0.2,
    #     neg_sampling_ratio=val_neg_sampling_ratio,
    #     # We don't want to add negative edges for the training set here because we
    #     # want them to vary for every epoch. Hence, we let the negative sampling
    #     # happen at the loader level for the training set. See below.
    #     add_negative_train_samples=False,
    #     edge_types=("question", "question_answer", "answer"),
    #     rev_edge_types=("answer", "rev_question_answer", "question"),
    # )
    #
    # train_data, val_data, test_data = transform(data)

    train_data, val_data, test_data = split_data(data, add_negative_train_samples=True)
    return {'train': train_data, 'val': val_data, 'test': test_data}


def build_link_neighbor_loaders(data, neg_sampling_ratio, num_neighbors, batch_size, val_neg_sampling_ratio=2.0):
    """
    Constructs 3 LinkNeighborLoaders (train, val, test) for the given data.

    "neg_sampling_ratio" is the negative sampling ratio for the training data,
    whereas "val_neg_sampling_ratio" is applied to the validation and test sets.
    """
    split_data_dict = random_link_split(data, val_neg_sampling_ratio)

    split_loaders = {}
    edge_type = ("question", "question_correct_answer", "answer")

    for split_name, split_data in split_data_dict.items():
        edge_label = split_data[edge_type].edge_label
        edge_label_index = (edge_type, split_data[edge_type].edge_label_index)

        is_train = split_name == 'train'

        split_loaders[split_name] = LinkNeighborLoader(
            split_data.clone().contiguous(),
            num_neighbors=num_neighbors,
            # Only the training set has negative sampling at the loader level; the
            # validation and test sets have negative sampling at the data split-level instead.
            # neg_sampling_ratio=neg_sampling_ratio if is_train else 0.0,
            neg_sampling_ratio=3.0,
            edge_label=edge_label,
            edge_label_index=edge_label_index,
            batch_size=batch_size
            # shuffle=is_train
            )

    return split_loaders
