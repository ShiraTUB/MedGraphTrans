import networkx as nx
import numpy as np
import torch
import os
import pickle
import openai

from KnowledgeExtraction.trie_structure import Trie
from KnowledgeExtraction.knowledge_extractor import KnowledgeExtractor
from torch_geometric.data import HeteroData
from subgraph import Graph, Subgraph

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def build_trie_from_kg(knowledge_graph: nx.Graph, save_path=None):
    """
    If you wish to use ConceptNet (or a similar KG dataset istead of a nx graph please use the function
    insert_dataset(dataset, embeddings) from the trie_structure file instead of insert_knowledge_graph(...)
    """
    trie = Trie()
    trie.insert_knowledge_graph(knowledge_graph, 'name', store_node_index=True)

    if save_path is not None:
        trie.save_trie(save_path)

    return trie


def build_trie_from_conceptnet(dataset, embeddings, save_path=None):
    trie = Trie()
    trie.insert_dataset(dataset, embeddings)

    if save_path is not None:
        trie.save_trie(save_path)

    return trie


def extract_knowledge_from_kg(question: str, answer_choices, trie: Trie, knowledge_graph: nx.Graph,
                              question_entities_list=None, answer_entities_dict=None):
    """
    :param question: the string which we want to enrich with the subgraph
    :param answer_choices: a list fo all possible answers to the question
    :param trie: the kg stored in a trie structure for more efficient search
    :param knowledge_graph: a nx graph, can be used to reconstruct the subgraph
        :param question_entities_list: an optional preprocessed list of entities extracted from the question. If none, every word (except from stopwords)of
        question will be processed
    :param answer_entities_dict: an optional preprocessed dict of entities extracted from each answer choice. If none, every word (except from stopwords)of
        question will be processed
    :return: The extracted knowledge (subgraph) stored in a nx graph
    """

    answers_entities_list = [entity for entities in answer_entities_dict.values() for entity in entities]

    # Find relevant subgraphs using DiaTransNet's trie_structure
    knowledge_extractor = KnowledgeExtractor(trie, question, entities=question_entities_list + answers_entities_list)

    if len(knowledge_extractor.entities) == 0:
        return None

    context_graph = initiate_question_graph(question, answer_choices, question_entities_list, answer_entities_dict)

    extracted_edges, extracted_edge_indices = knowledge_extractor.extract_subgraph_from_query(n_hops=2, k=10)
    if extracted_edges is None:
        return None

    context_graph['knowledge_knowledge_edges'] = extracted_edges

    return context_graph

    # """We offer two ways to construct the subgraph:
    #     1. if node indices were stored in the trie they can be used to extract the subgraph directly from the kg
    #     2. if not, an elementary subgraph is created from the extracted graph edges. Please make adjustments
    #         according to your kg and its attributes."""
    # if extracted_edge_indices is not None:
    #     node_indices = np.unique(np.asarray(np.concatenate(extracted_edge_indices)))
    #     subgraph = knowledge_graph.subgraph(node_indices.tolist())
    #
    # else:
    #     subgraph = nx.Graph()
    #     for index, (source, target, relation) in enumerate(extracted_edges):
    #         if not subgraph.has_node(source):
    #             subgraph.add_node(source)
    #         if not subgraph.has_node(target):
    #             subgraph.add_node(target)
    #         if not subgraph.has_edge(source, target):
    #             subgraph.add_edge(source, target, relation={relation})
    #
    # return subgraph, extracted_edges


def convert_nx_to_hetero_data(graph: nx.Graph, node_types: list, relation_types: list,
                              meta_relation_dict: dict) -> HeteroData:
    """
    :param graph: nx graph to be transformed into hetero data
    :param node_types: a list of all possible node types
    :param relation_types: a list of all possible relation types
    :param meta_relation_dict: a dictionary that maps each relation type to a meta relation in the form (source type, relation type, target type)
    :return: the hetero data crated from the graph
    """
    data = HeteroData()
    for node_type in node_types:

        node_type_embeddings = [np.squeeze(data['embedding']) for node, data in graph.nodes(data=True) if
                                data['type'] == node_type]
        if len(node_type_embeddings) > 0:
            data[node_type].x = torch.stack(node_type_embeddings, dim=0).type("torch.FloatTensor")

            # TODO: find the right masking strategy

    for relation_type in relation_types:
        meta_rel_type = meta_relation_dict[relation_type]

        source_target_indices = np.asarray([(source, target) for (source, target, data) in graph.edges(data=True) if
                                            data['relation'] == relation_type])

        if len(source_target_indices) > 0:
            edge_index_feats = [torch.tensor(source_target_indices[:, 0]), torch.tensor(source_target_indices[:, 1])]

            data[meta_rel_type[0], meta_rel_type[1], meta_rel_type[2]].edge_index = torch.stack(
                edge_index_feats, dim=0)

    return data


def initiate_question_graph(question: str, answer_choices: [str], question_entities, answer_entities_dict):
    graph_data = {}

    # question - knowledge edges

    question_entities_nodes = np.char.array(question_entities)
    question_entities_length = len(question_entities_nodes)

    question_node = np.char.array(question)
    question_node = np.resize(question_node, question_entities_length)

    question_relation = np.char.array("knowledge")
    question_relation = np.resize(question_relation, question_entities_length)
    question_edges = np.char.array([question_node, question_entities_nodes, question_relation]).T
    graph_data['question_knowledge_edges'] = question_edges
    graph_data['question_knowledge_nodes'] = question_entities_nodes

    # answer_choice - knowledge edges

    for index, answer_choice_entities in enumerate(answer_entities_dict.values()):
        answer_entities_nodes = np.char.array(answer_choice_entities)
        answer_entities_length = len(answer_entities_nodes)

        answer_node = np.char.array(answer_choices[index])
        answer_node = np.resize(answer_node, answer_entities_length)

        answer_relation = np.char.array("knowledge")
        answer_relation = np.resize(answer_relation, answer_entities_length)
        answer_edges = np.char.array([answer_node, answer_entities_nodes, answer_relation]).T
        graph_data['answer_knowledge_edges'] = answer_edges
        graph_data['answer_knowledge_nodes'] = answer_entities_nodes

    # question - answer_choice edges

    answer_choices_nodes = np.char.array(answer_choices)
    answer_choices_length = len(answer_choices_nodes)

    question_node = np.char.array(question)
    question_node = np.resize(question_node, answer_choices_length)

    relation = np.char.array("answer_choice")
    relation = np.resize(relation, answer_choices_length)
    edges = np.char.array([question_node, answer_choices_nodes, relation]).T
    graph_data['question_answer_edges'] = edges
    graph_data['answer_nodes'] = answer_choices_nodes

    return graph_data


def process_raw_graph_data(graph_data, processed_dir=None, filename=None, index=None):
    graph = create_graph_from_data(graph_data)

    if processed_dir is not None and filename is not None and index is not None:
        save_graph(graph, processed_dir, filename, index)


def create_graph_from_data(graph_data):
    graph = Graph()

    question_knowledge_edges = graph_data['question_knowledge_edges']
    answer_knowledge_edges = graph_data['answer_knowledge_edges']
    question_answer_edges = graph_data['question_answer_edges']
    knowledge_knowledge_edges = graph_data['knowledge_knowledge_edges']

    question_node, answer_nodes, knowledge_nodes = extract_nodes(question_knowledge_edges, answer_knowledge_edges,
                                                                 knowledge_knowledge_edges)

    question_string2int, question_int2string = get_converters(question_node)
    answer_string2int, answer_int2string = get_converters(answer_nodes)
    know_string2int, know_int2string = get_converters(knowledge_nodes)

    question_node, answer_nodes, knowledge_nodes = tokenize_nodes(question_node, answer_nodes, knowledge_nodes)

    question_knowledge, answer_knowledge, question_answer, knowledge_knowledge = initialize_edges_tensors(
        question_knowledge_edges, answer_knowledge_edges,
        question_answer_edges, knowledge_knowledge_edges, question_string2int,
        answer_string2int, know_string2int)

    subgraph = Subgraph(question_node, answer_nodes, knowledge_nodes, question_knowledge, answer_knowledge, question_answer, knowledge_knowledge)
    graph.insert_subgraph(subgraph)

    return graph


def extract_nodes(question_knowledge_edges, answer_knowledge_edges, knowledge_knowledge_edges):
    question_node = np.unique(question_knowledge_edges[0])
    answer_nodes = np.unique(answer_knowledge_edges[0])

    knowledge_nodes = []

    if question_knowledge_edges is not None:
        knowledge_nodes.extend(np.unique(question_knowledge_edges[:, 1]))
    if answer_knowledge_edges is not None:
        knowledge_nodes.extend(np.unique(answer_knowledge_edges[:, 1]))
    if knowledge_knowledge_edges is not None:
        knowledge_nodes.extend(np.unique(np.concatenate(
            (np.char.array([[ke[0].replace("_", " "), ke[1].replace("_", " ")] for ke in knowledge_knowledge_edges])
             ))))

    if len(knowledge_nodes) == 0:
        knowledge_nodes = None

    return question_node, answer_nodes, knowledge_nodes


def tokenize_nodes(question_node, answer_nodes, knowledge_nodes):
    question_node = openai.Embedding.create(input=[question_node], model="text-embedding-ada-002")['data'][0][
        'embedding']

    answer_nodes_embeddings = []
    for answer in answer_nodes:
        answer_tensor = openai.Embedding.create(input=[answer], model="text-embedding-ada-002")['data'][0]['embedding']
        answer_nodes_embeddings.extend(answer_tensor)

    answer_nodes = torch.stack(answer_nodes_embeddings, dim=0).type("torch.FloatTensor")

    knowledge_nodes_embeddings = []
    for knowledge in knowledge_nodes:
        knowledge_tensor = openai.Embedding.create(input=[knowledge], model="text-embedding-ada-002")['data'][0][
            'embedding']
        knowledge_nodes_embeddings.extend(knowledge_tensor)

    knowledge_nodes = torch.stack(knowledge_nodes_embeddings, dim=0).type("torch.FloatTensor")

    return question_node, answer_nodes, knowledge_nodes


def get_converters(str_list):
    string2int = {}
    count = 0

    for string in str_list:
        if string not in string2int:
            string2int[string] = count
            count += 1

    int2string = {}

    for index, key in enumerate(string2int.keys()):
        int2string[index] = key

    return string2int, int2string


def convert_nodes(nodes, converter):
    convertet_nodes = torch.ones(size=nodes.shape)

    for index, node in enumerate(nodes):
        convertet_nodes[index] = converter[node]

    return convertet_nodes


def convert_edges(edges, head_converter, tail_converter):
    convertet_edges = torch.ones(size=edges.shape)

    for index, edge in enumerate(edges):
        head, tail = edge
        convertet_edges[index][0] = head_converter[head]
        convertet_edges[index][1] = tail_converter[tail]

    return convertet_edges


def initialize_edges_tensors(question_knowledge_edges, answer_knowledge_edges,
                             question_answer_edges, knowledge_knowledge_edges, question_string2int,
                             answer_string2int, know_string2int):
    question_knowledge_edges = convert_edges(question_knowledge_edges[0:2], question_string2int,
                                             know_string2int).T if question_knowledge_edges.ndim > 1 else convert_edges(
        question_knowledge_edges[None, 0:2], question_string2int, know_string2int).T

    answer_knowledge_edges = convert_edges(answer_knowledge_edges[:, 0:2], answer_string2int,
                                           know_string2int).T if answer_knowledge_edges.ndim > 1 else convert_edges(
        answer_knowledge_edges[None, 0:2], answer_string2int, know_string2int).T

    if knowledge_knowledge_edges is not None:
        knowledge_knowledge_edges = convert_edges(knowledge_knowledge_edges[:, 0:2], know_string2int, know_string2int).T
    else:
        knowledge_knowledge_edges = torch.empty(size=(2, 0))

    question_answer_edges = convert_edges(question_answer_edges[:, 0:2], question_string2int, answer_string2int).T

    return question_knowledge_edges, answer_knowledge_edges, knowledge_knowledge_edges, question_answer_edges


def save_graph(graph, processed_dir, filename, index):
    pickle.dump(graph, open(os.path.join(processed_dir, f'{filename}_{index}.p'), "wb"))
