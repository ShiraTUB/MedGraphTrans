import openai
import torch

from torch import Tensor
from collections import deque
from torch_geometric.data import HeteroData
from torch.nn import ParameterDict
from typing import Dict, List, Optional, Tuple
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, NodeType, SparseTensor
from torch_geometric.utils import is_sparse, to_edge_index

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


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
        edge_weights_dict: Optional[Dict[EdgeType, Tensor]] = None,
) -> Tuple[Adj, Optional[Tensor], Optional[Tensor], Dict[EdgeType, int]]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes.

    Args:
        edge_weights_dict:
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
    edge_weights: List[Tensor] = []
    edge_offset_dict = {}
    for edge_type, src_offset in src_offset_dict.items():
        edge_index = edge_index_dict[edge_type]
        edge_offset_dict[edge_type] = edge_index.size(1)
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

        if edge_weights_dict is not None:
            if isinstance(edge_weights_dict, ParameterDict):
                edge_weight = edge_weights_dict['__'.join(edge_type)]
            else:
                edge_weight = edge_weights_dict[edge_type]
            if edge_weight.size(0) != edge_index.size(1):
                edge_weight = edge_weight.expand(edge_index.size(1), -1)
            edge_weights.append(edge_weight)

    edge_index = torch.cat(edge_indices, dim=1)

    edge_attr: Optional[Tensor] = None
    if edge_attr_dict is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)

    edge_weight: Optional[Tensor] = None
    if edge_weights_dict is not None:
        edge_weight = torch.cat(edge_attrs, dim=0)

    if is_sparse_tensor:
        # TODO Add support for `SparseTensor.sparse_sizes()`.
        edge_index = SparseTensor(
            row=edge_index[1],
            col=edge_index[0],
            value=edge_attr,
        )

    return edge_index, edge_attr, edge_weight, edge_offset_dict


def find_most_relevant_nodes(batch, z_dict, question_nodes_embedding, knowledge_nodes_uid_dict, prime_gk, similarity_threshold=0.65):
    relevant_nodes_list = []
    relevant_nodes_names, relevant_nodes_types = [], []
    for node_type, nodes_uids in knowledge_nodes_uid_dict.items():
        node_indices = [torch.where(batch[node_type].node_uid == x)[0][0] for x in nodes_uids]
        node_embeddings = torch.index_select(z_dict[node_type], 0, torch.tensor(node_indices))

        # Calculate the distance between the node embedding and the central node embedding
        distances = torch.norm(question_nodes_embedding.repeat(node_embeddings.size(0), 1) - node_embeddings, p=2, dim=1)
        threshold_dist_indices = torch.where(distances <= similarity_threshold)
        relevant_nodes_uids = nodes_uids[threshold_dist_indices]
        for index in relevant_nodes_uids:
            node_type = prime_gk.nodes[index.item()]['type']
            node_name = prime_gk.nodes[index.item()]['name']
            relevant_nodes_list.append(f'The {node_type} {node_name}')

    return relevant_nodes_list


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


def find_subgraph_bfs(graph: HeteroData, start_node_index: int, start_node_type: str):
    visited_nodes = set()
    # subgraph_nodes = set()
    subgraph_dict = {}
    queue = deque([(start_node_index, start_node_type)])
    while queue:
        current_node, node_type = queue.popleft()
        visited_nodes.add((current_node, node_type))
        if node_type != 'question' and node_type != 'answer':
            if node_type not in subgraph_dict:
                subgraph_dict[node_type] = []
            subgraph_dict[node_type].append(graph[node_type].node_uid[current_node].item())
            # subgraph_nodes.add((graph[node_type].node_uid[current_node].item(), node_type))

        # Iterate over all edge types
        for edge_type in graph.edge_types:
            # Check if the current node type is part of the edge type
            if node_type in edge_type:

                # Determine the node type at the other end of the edge
                other_node_type = edge_type[2] if node_type == edge_type[0] else edge_type[0]

                # Get the edge index for the current edge type
                edge_index = graph[edge_type].edge_index

                # Find the neighbors
                neighbors = edge_index[1][edge_index[0] == current_node] if node_type == edge_type[0] else edge_index[0][edge_index[1] == current_node]
                for neighbor in neighbors:
                    if (neighbor.item(), other_node_type) not in visited_nodes:
                        visited_nodes.add((neighbor.item(), other_node_type))
                        if other_node_type != 'question' and other_node_type != 'answer':
                            if other_node_type not in subgraph_dict:
                                subgraph_dict[other_node_type] = []
                            subgraph_dict[other_node_type].append(graph[other_node_type].node_uid[neighbor].item())
                            # subgraph_nodes.add((graph[other_node_type].node_uid[neighbor].item(), other_node_type))
                        queue.append((neighbor.item(), other_node_type))

    for n_type in subgraph_dict.keys():
        subgraph_dict[n_type] = torch.unique(torch.tensor(subgraph_dict[n_type]))

    return subgraph_dict


def compute_llm_confidence_diff(confidence_without_context, confidence_with_context):
    # todo: work on logic

    return confidence_with_context - confidence_without_context


def query_chatbot(prompt, output_instructions):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt + output_instructions},
            ],
            request_timeout=15,
            n=1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed with an exception: {e}")
        return -1
