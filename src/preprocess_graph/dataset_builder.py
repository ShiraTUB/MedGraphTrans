import os.path
import pickle
import networkx as nx

from src.preprocess_graph.build_subgraph import convert_nx_to_hetero_data


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

def fix_graph(nx_graph):
    question_node = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'question'][0]
    answer_nodes = [node for node, attr in nx_graph.nodes(data=True) if attr.get('type') == 'answer']

    for answer_node in answer_nodes:
        if nx_graph.has_edge(question_node, answer_node):
            nx_graph[question_node][answer_node]['relation'] = "question_correct_answer"
        else:
            nx_graph.add_edge(question_node, answer_node, relation="question_wrong_answer")

    return nx_graph


train_graph_data_l = build_data_list('../../datasets/raw_graph_dataset/train')
val_graph_data_l = build_data_list('../../datasets/raw_graph_dataset/validation')
# test_graph_data_l = build_data_list('../../datasets/raw_graph_dataset/test')

fixed_train_graph_l = []
for graph in train_graph_data_l:
    fixed_train_graph_l.append(fix_graph(graph))

fixed_val_graph_l = []
for graph in val_graph_data_l:
    fixed_val_graph_l.append(fix_graph(graph))

merged_data_list = fixed_train_graph_l + fixed_val_graph_l

merged_graph = build_merged_graphs_raw_dataset(merged_data_list)

merged_hetero_data = convert_nx_to_hetero_data(merged_graph, ('question', 'question_correct_answer', 'answer'))

pickle.dump(merged_hetero_data, open(os.path.join('../../datasets/merged_hetero_dataset/processed_graph_1000_train_val_masked_with_edge_uids_and neg_edges.pickle'), 'wb'))

print('end')
