import os.path
import pickle
from torch_geometric.data import HeteroData
import torch
from build_subgraph import contains_tensor
import networkx as nx


def build_data_list(root_dir: str):
    data_list = []
    file_names_list = os.listdir(root_dir)
    for file_name in file_names_list:
        path = os.path.join(root_dir, file_name)
        with open(path, 'rb') as f:
            hetero_data = pickle.load(f)
            data_list.append(hetero_data)
    return data_list


def build_merged_graphs_raw_dataset(graphs_list):
    merged_graph = nx.Graph()

    # Merge all graphs into the merged_graph
    for graph in graphs_list:
        merged_graph.add_nodes_from(graph.nodes(data=True))
        merged_graph.add_edges_from(graph.edges(data=True))

    return merged_graph

# At this point, merged_graph contains all the nodes and edges from graph1, graph2, and graph3


# def merge_hetero_data_list(hetero_data_list: [HeteroData]):
#     # Initialize an empty HeteroData object
#     large_hetero_data = HeteroData()
#
#     # Initialize offsets for each node type
#     node_offsets = {node_type: 0 for node_type in large_hetero_data.node_types}
#
#     for hetero_data in hetero_data_list:
#         for node_type in hetero_data.node_types:
#             # Concatenate node features for each node type
#             if node_type in large_hetero_data.node_types:
#                 large_hetero_data[node_type].x = torch.cat([large_hetero_data[node_type].x, hetero_data[node_type].x], dim=0)
#             else:
#                 large_hetero_data[node_type].x = hetero_data[node_type].x.clone()
#                 node_offsets[node_type] = 0
#
#         for edge_type in hetero_data.edge_types:
#             # Adjust edge indices based on the current offset
#             src_node_type, _, dst_node_type = edge_type
#             src_offset = node_offsets[src_node_type]
#             dst_offset = node_offsets[dst_node_type]
#
#             # Adjust edge indices with the respective offsets
#             adjusted_edge_index = hetero_data[edge_type].edge_index.clone()
#             adjusted_edge_index[0] += src_offset
#             adjusted_edge_index[1] += dst_offset
#
#             # Concatenate edge indices and edge attributes for each edge type
#             if edge_type in large_hetero_data.edge_types:
#                 large_hetero_data[edge_type].edge_index = torch.cat([large_hetero_data[edge_type].edge_index, adjusted_edge_index], dim=1)
#                 # large_hetero_data[edge_type].edge_attr = torch.cat([large_hetero_data[edge_type].edge_attr, hetero_data[edge_type].edge_attr], dim=0)
#             else:
#                 large_hetero_data[edge_type].edge_index = adjusted_edge_index
#                 # large_hetero_data[edge_type].edge_attr = hetero_data[edge_type].edge_attr.clone()
#             # Update the offset for this node type
#
#         for node_type in hetero_data.node_types:
#             node_offsets[node_type] += hetero_data[node_type].num_nodes
#
#             # If necessary, update any other attributes like edge labels, etc.
#
#     return large_hetero_data
#
#
# def remove_duplicates(hetero_data):
#     # Dictionary for mapping old indices to new indices for each node type
#     index_map = {}
#
#     # Iterate over node types to remove duplicates
#     for node_type in hetero_data.node_types:
#         node_features = hetero_data[node_type].x.clone()
#         unique_tensors, unique_indices = node_features.unique(dim=0, return_inverse=True)
#
#
#         # nique_features = features[unique_indices == torch.arange(features.size(0))[:, None].expand_as(unique_indices)]
#         hetero_data[node_type].x = unique_tensors
#         # Store the mapping of old to new indices
#         index_map[node_type] = unique_indices
#
#     # Update the edge indices for each edge type
#     for edge_type in hetero_data.edge_types:
#         edge_index = hetero_data[edge_type].edge_index
#         for i, node_type in enumerate(edge_type[0], edge_type[2]):  # Iterate over both source and target node types
#             if node_type in index_map:  # Check if this node type had duplicates removed
#                 new_indices = index_map[node_type]
#                 # Update the edge indices
#                 edge_index[i, :] = new_indices[edge_index[i, :]]
#                 hetero_data[edge_type].edge_index = edge_index


hetero_data_l = build_data_list('../../datasets/raw_graph_dataset/train')
merged_graph = build_merged_graphs_raw_dataset(hetero_data_l)
pickle.dump(merged_graph, open(os.path.join('../../datasets/raw_graph_dataset/train/merged_graph_500.pickle'), 'wb'))

print('end')
