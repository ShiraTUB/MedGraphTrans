import networkx as nx
import numpy as np
import torch
import os
import pickle
import openai
import random

from KnowledgeExtraction.trie_structure import Trie
from KnowledgeExtraction.knowledge_extractor import KnowledgeExtractor
from torch_geometric.data import HeteroData
from src.preprocess_graph.subgraph import Graph

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


def extract_knowledge_from_kg(question: str, trie: Trie, question_entities_list=None, answer_entities_list: list = None):
    """
    :param question: the string which we want to enrich with the subgraph
    :param trie: the kg stored in a trie structure for more efficient search
    :param question_entities_list: an optional preprocessed list of entities extracted from the question. If none, every word (except from stopwords)of
        question will be processed
    :param answer_entities_list: an optional preprocessed list of entities extracted from each answer choice. If none, every word (except from stopwords)of
        question will be processed
    :return: The extracted knowledge (subgraph) stored in a nx graph
    """

    answers_entities_list = [entity for entities in answer_entities_list for entity in entities]

    # Find relevant subgraphs using DiaTransNet's trie_structure
    knowledge_extractor = KnowledgeExtractor(trie, question, entities=question_entities_list + answers_entities_list)

    if len(knowledge_extractor.entities) == 0:
        return None, None

    extracted_edges, extracted_edge_indices = knowledge_extractor.extract_subgraph_from_query(n_hops=2, k=10)

    if extracted_edges is None:
        return None, None

    return extracted_edges, extracted_edge_indices

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


def convert_nx_to_hetero_data(graph: nx.Graph, node_types: list, relation_types: list, meta_relation_dict: dict) -> HeteroData:
    """
    :param graph: nx graph to be transformed into hetero data
    :param node_types: a list of all possible node types
    :param relation_types: a list of all possible relation types
    :param meta_relation_dict: a dictionary that maps each relation type to a meta relation in the form (source type, relation type, target type)
    :return: the hetero data crated from the graph
    """
    data = HeteroData()

    for node_type in node_types:

        node_type_embeddings = [np.squeeze(data['embedding']) for node, data in graph.nodes(data=True) if data['type'] == node_type]

        if len(node_type_embeddings) > 0:
            # TODO: find the right masking strategy
            data[node_type].x = torch.stack(node_type_embeddings, dim=0).type("torch.FloatTensor")

    for relation_type in relation_types:
        meta_rel_type = meta_relation_dict[relation_type]

        source_target_indices = np.asarray([(source, target) for (source, target, data) in graph.edges(data=True) if data['relation'] == relation_type])

        if len(source_target_indices) > 0:
            edge_index_feats = [torch.tensor(source_target_indices[:, 0]), torch.tensor(source_target_indices[:, 1])]

            data[meta_rel_type[0], meta_rel_type[1], meta_rel_type[2]].edge_index = torch.stack(
                edge_index_feats, dim=0)

    return data


def initiate_question_graph_dict(question: str, answer_choices: [str], question_entities, answer_entities_dict) -> dict:
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
    graph_data['question_node'] = question_node
    question_node = np.resize(question_node, answer_choices_length)

    relation = np.char.array("answer_choice")
    relation = np.resize(relation, answer_choices_length)
    edges = np.char.array([question_node, answer_choices_nodes, relation]).T
    graph_data['question_answer_edges'] = edges
    graph_data['answer_nodes'] = answer_choices_nodes

    return graph_data


def initiate_question_graph(graph: nx.Graph, question: str, answer_choices: [str], question_entities_indices_list: list, answer_entities_dict: dict, prime_kg: nx.Graph) -> nx.Graph:
    question_embeddings = openai.Embedding.create(input=[question], model="text-embedding-ada-002")['data'][0]['embedding']
    question_index = random.randint(10 ** 9, (10 ** 10) - 1)

    graph.add_node(question_index, embedding=question_embeddings, type="question", index=question_index, name=question)

    for question_entity_index in question_entities_indices_list:
        target_node = prime_kg.nodes[question_entity_index]
        graph.add_node(question_entity_index, **target_node)
        graph.add_edge(question_index, question_entity_index, relation="question_knowledge")

    for answer_choice in answer_choices:
        answer_embeddings = openai.Embedding.create(input=[answer_choice], model="text-embedding-ada-002")['data'][0]['embedding']

        answer_index = random.randint(10 ** 9, (10 ** 10) - 1)
        graph.add_node(answer_index, embedding=answer_embeddings, type="answer", index=answer_index, name=answer_choice)
        graph.add_edge(question_index, answer_index, relation="question_answer")

        for answer_entity_index in answer_entities_dict[answer_choice]:
            target_node = prime_kg.nodes[answer_entity_index]
            graph.add_node(answer_entity_index, **target_node)
            graph.add_edge(answer_index, answer_entity_index, relation="answer_knowledge")

    return graph


def expand_graph_with_knowledge(graph: nx.Graph, extracted_edge_indices: [list], prime_kg: nx.Graph) -> nx.Graph:
    for source, target in extracted_edge_indices:
        source_node = prime_kg.nodes[source]
        target_node = prime_kg.nodes[target]
        relation = prime_kg[source][target][0]['relation']

        graph.add_node(source, **source_node)
        graph.add_node(target, **target_node)
        graph.add_edge(source, target, relation=relation)

    return graph


def process_raw_graph_data(graph_data):
    graph = Graph()

    question_knowledge_edges = graph_data['question_knowledge_edges']
    answer_knowledge_edges = graph_data['answer_knowledge_edges']
    question_answer_edges = graph_data['question_answer_edges']
    knowledge_knowledge_edges = graph_data['knowledge_knowledge_edges']
    answer_nodes = list(graph_data['answer_nodes'])
    question_node = list(np.unique(question_knowledge_edges[:, 0]))

    knowledge_nodes_type_dict = extract_knowledge_nodes(question_knowledge_edges, answer_knowledge_edges, knowledge_knowledge_edges)
    knowledge_nodes = list(knowledge_nodes_type_dict.keys())

    question_node_string_to_int_mapping = get_converters(question_node)
    answer_nodes_string_to_int_mapping = get_converters(answer_nodes)
    knowledge_nodes_string_to_int_mapping = get_converters(knowledge_nodes)

    question_node_tensor, answer_nodes_tensor, knowledge_nodes_tensor = tokenize_nodes(question_node[0], answer_nodes, knowledge_nodes)

    question_knowledge_tensor, answer_knowledge_tensor, question_answer_tensor, knowledge_knowledge_tensor = initialize_edges_tensors(
        question_knowledge_edges,
        answer_knowledge_edges,
        question_answer_edges,
        knowledge_knowledge_edges,
        question_node_string_to_int_mapping,
        answer_nodes_string_to_int_mapping,
        knowledge_nodes_string_to_int_mapping
    )

    # subgraph = Subgraph(question_node_tensor, answer_nodes_tensor, knowledge_nodes_tensor, question_knowledge_tensor, answer_knowledge_tensor, question_answer_tensor,
    #                     knowledge_knowledge_tensor)
    # graph.insert_subgraph(subgraph)

    return graph


def extract_knowledge_nodes(question_knowledge_edges, answer_knowledge_edges, knowledge_knowledge_edges):
    knowledge_nodes_type_dict = {}

    for edge in knowledge_knowledge_edges:
        source, target, type_ = edge
        source_type = type_.split('_')[0]
        target_type = type_.split('_')[1]

        # add source and target nodes to the dictionary
        knowledge_nodes_type_dict[source] = source_type
        knowledge_nodes_type_dict[target] = target_type

    for node in list(np.unique(question_knowledge_edges[:, 1])) + list(np.unique(answer_knowledge_edges[:, 1])):
        if node not in knowledge_nodes_type_dict:
            knowledge_nodes_type_dict[node] = 'knowledge'

    return knowledge_nodes_type_dict


def tokenize_nodes(question_node, answer_nodes, knowledge_nodes):
    question_node_tensor = torch.tensor(openai.Embedding.create(input=[question_node], model="text-embedding-ada-002")['data'][0]['embedding'])

    answer_nodes_embeddings = []

    for answer in answer_nodes:
        answer_tensor = openai.Embedding.create(input=[answer], model="text-embedding-ada-002")['data'][0]['embedding']
        answer_nodes_embeddings.append(answer_tensor)

    answer_nodes_tensor = torch.tensor(answer_nodes_embeddings)

    knowledge_nodes_embeddings = []

    for knowledge in knowledge_nodes:
        knowledge_tensor = openai.Embedding.create(input=[knowledge], model="text-embedding-ada-002")['data'][0]['embedding']
        knowledge_nodes_embeddings.append(knowledge_tensor)

    knowledge_nodes_tensor = torch.tensor(knowledge_nodes_embeddings)

    return question_node_tensor, answer_nodes_tensor, knowledge_nodes_tensor


def get_converters(str_list):
    string2int = {}
    count = 0

    for string in str_list:
        if string not in string2int:
            string2int[string] = count
            count += 1

    return string2int


def convert_nodes(nodes, converter):
    convertet_nodes = torch.ones(size=nodes.shape)

    for index, node in enumerate(nodes):
        convertet_nodes[index] = converter[node]

    return convertet_nodes


def convert_edges(edges, source_converter, target_converter):
    converted_edges = torch.ones(size=edges.shape)

    for index, edge in enumerate(edges):
        source, target = edge
        converted_edges[index][0] = source_converter[source]
        converted_edges[index][1] = target_converter[target]

    return converted_edges


def initialize_edges_tensors(question_knowledge_edges, answer_knowledge_edges,
                             question_answer_edges, knowledge_knowledge_edges, question_node_string_to_int_mapping,
                             answer_nodes_string_to_int_mapping, knowledge_nodes_string_to_int_mapping):
    if question_knowledge_edges.ndim > 1:
        question_knowledge_edges_tensor = convert_edges(question_knowledge_edges[:, :2], question_node_string_to_int_mapping, knowledge_nodes_string_to_int_mapping).T
    else:
        question_knowledge_edges_tensor = convert_edges(question_knowledge_edges[None, :, :2], question_node_string_to_int_mapping, knowledge_nodes_string_to_int_mapping).T

    if answer_knowledge_edges.ndim > 1:
        answer_knowledge_edges_tensor = convert_edges(answer_knowledge_edges[:, :2], answer_nodes_string_to_int_mapping, knowledge_nodes_string_to_int_mapping).T
    else:
        answer_knowledge_edges_tensor = convert_edges(answer_knowledge_edges[None, :, :2], answer_nodes_string_to_int_mapping, knowledge_nodes_string_to_int_mapping).T

    if knowledge_knowledge_edges is not None:
        knowledge_knowledge_edges_tensor = convert_edges(knowledge_knowledge_edges[:, :2], knowledge_nodes_string_to_int_mapping, knowledge_nodes_string_to_int_mapping).T
    else:
        knowledge_knowledge_edges_tensor = torch.empty(size=(2, 0))

    question_answer_edges_tensor = convert_edges(question_answer_edges[:, :2], question_node_string_to_int_mapping, answer_nodes_string_to_int_mapping).T

    return question_knowledge_edges_tensor, answer_knowledge_edges_tensor, knowledge_knowledge_edges_tensor, question_answer_edges_tensor


def save_graph(graph, processed_dir, filename, index):
    pickle.dump(graph, open(os.path.join(processed_dir, f'{filename}_{index}.p'), "wb"))
