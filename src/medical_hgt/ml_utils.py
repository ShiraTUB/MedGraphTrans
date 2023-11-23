import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch.nn import ParameterDict
from typing import Dict, List, Optional, Tuple
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, NodeType, SparseTensor
from torch_geometric.utils import is_sparse, to_edge_index


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


def construct_bipartite_edge_index(
        edge_index_dict: Dict[EdgeType, Adj],
        src_offset_dict: Dict[EdgeType, int],
        dst_offset_dict: Dict[NodeType, int],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
        relevant_edge_weights_indices_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_weight_dict: Optional[ParameterDict] = None
) -> Tuple[Adj, Optional[Tensor], Optional[Tensor]]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes.

    Args:
        edge_weight_dict: New parameter for edge weights
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
            dictionary holding graph connectivity information for each
            individual edge type, either as a :class:`torch.Tensor` of
            shape :obj:`[2, num_edges]` or a
            :class:`torch_sparse.SparseTensor`.
        src_offset_dict (Dict[Tuple[str, str, str], int]): A dictionary of
            offsets to apply to the source node type for each edge type.
        dst_offset_dict (Dict[str, int]): A dictionary of offsets to apply for
            destination node types.
        edge_attr_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
            dictionary holding edge features for each individual edge type.
            (default: :obj:`None`)
    """
    is_sparse_tensor = False
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    edge_weights: List[Tensor] = []  # List to store concatenated edge weights

    for edge_type, src_offset in src_offset_dict.items():
        edge_index = edge_index_dict[edge_type]
        dst_offset = dst_offset_dict[edge_type[-1]]

        # TODO Add support for SparseTensor w/o converting.
        is_sparse_tensor = isinstance(edge_index, SparseTensor)

        if is_sparse(edge_index):
            edge_index, _ = to_edge_index(edge_index)
            edge_index = edge_index.flip([0])
        else:
            edge_index = edge_index.clone()

        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_indices.append(edge_index)

        if edge_attr_dict is not None:
            if isinstance(edge_attr_dict, ParameterDict):
                edge_attr = edge_attr_dict['__'.join(edge_type)]
            else:
                edge_attr = edge_attr_dict[edge_type]

            if edge_attr.size(0) != edge_index.size(1):
                edge_attr = edge_attr.expand(edge_index.size(1), -1)

            edge_attrs.append(edge_attr)

        # Handle edge weights
        if edge_weight_dict is not None and '__'.join(edge_type) in edge_weight_dict:
            # edge_weight = edge_weight_dict['__'.join(edge_type)][0][relevant_edge_weights_indices_dict['__'.join(edge_type)]]

            index = relevant_edge_weights_indices_dict['__'.join(edge_type)]
            selected_edge_weight = torch.index_select(
                edge_weight_dict['__'.join(edge_type)][0], 0, index
            )

            # if edge_weight.size(0) != edge_index.size(1):
            #     edge_weight = edge_weight.expand(edge_index.size(1), -1)

            edge_weights.append(selected_edge_weight)

    edge_index = torch.cat(edge_indices, dim=1)

    edge_attr: Optional[Tensor] = None

    if edge_attr_dict is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)

    edge_weight: Optional[Tensor] = None

    if edge_weight_dict is not None:
        edge_weight = torch.cat(edge_weights, dim=0)
        edge_weight = torch.stack((edge_weight, edge_weight), dim=1)

    if is_sparse_tensor:
        # TODO Add support for `SparseTensor.sparse_sizes()`.
        edge_index = SparseTensor(
            row=edge_index[1],
            col=edge_index[0],
            value=edge_attr,
        )

    return edge_index, edge_attr, edge_weight  # Return edge weights as well


def decode_edge_weights(edge_weights_dict: dict, edge_index_dict: dict, all_edges_dict: dict, prime_kg: nx.Graph, relevancy_threshold: float = 0.7):
    # todo WIP
    for edge_type, edge_type_weights in edge_weights_dict.items():
        relevant_edges_indices = torch.where(edge_type_weights.squeeze() >= relevancy_threshold)[0]
        relevant_edges_uid = all_edges_dict[tuple(edge_type.split('__'))][relevant_edges_indices]
        relevant_edges = edge_index_dict[relevant_edges_indices]
        relevant_source_node_indices, relevant_target_node_indices = relevant_edges[0], relevant_edges[1]

        relevant_edges = [prime_kg.edges[s, t] for s, t in zip(relevant_source_node_indices, relevant_target_node_indices)]

        return relevant_edges


def compute_weight(node_type: str, node_index: int, edge_index_dict: dict, edge_weights_dict: dict):
    weights = []
    edge_types = set([edge_type for edge_type in edge_weights_dict.keys() if edge_type.split('__')[-1] == node_type])
    for edge_type in edge_types:
        relevant_indices = torch.where(edge_index_dict[tuple(edge_type.split('__'))][1] == node_index)[0]
        selected_edge_weight = torch.index_select(
            edge_weights_dict[edge_type], 0, relevant_indices
        )
        for w in selected_edge_weight:
            if w.numel() > 0:
                if bool(w) != 0:
                    weights.append(w)

    if len(weights) > 0:
        return torch.prod(torch.stack(weights, dim=0))
    else:
        return 1
