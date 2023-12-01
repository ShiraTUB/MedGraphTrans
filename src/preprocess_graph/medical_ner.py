import networkx as nx
import numpy as np
import openai
import torch
from config import OPENAI_API_KEY
from transformers import pipeline

openai.api_key = OPENAI_API_KEY


def medical_ner(query: str, knowledge_graph_embeddings: np.ndarray, node_indices_list: list, prime_kg: nx.Graph) -> (list, list):
    tokens = classify_tokens(query)
    clean_tokens_list = clean_tokens(tokens)
    entities_embeddings_list = embed_tokens(clean_tokens_list)

    if len(entities_embeddings_list) == 0:
        return [], []

    closest_entities, closest_entities_indices = find_closest_nodes(entities_embeddings_list, knowledge_graph_embeddings, node_indices_list, prime_kg)

    return closest_entities, closest_entities_indices


def classify_tokens(query: str) -> list:
    token_classifier = pipeline("token-classification", model="ukkendane/bert-medical-ner")
    return token_classifier(query)


def clean_tokens(tokens: list) -> list:
    clean_tokens = []
    current_entity = ""
    current_entity_type = ""

    for token in tokens:
        if token['entity'].startswith('B_'):
            if current_entity and current_entity_type != 'person' and current_entity_type != 'pronoun':
                clean_tokens.append(current_entity)
            current_entity = token['word']
            current_entity_type = token['entity'][2:]
        elif token['entity'].startswith('I_') and '##' in token['word']:
            current_entity += token['word'].replace('##', '')
        elif token['entity'].startswith('I_'):
            current_entity += " " + token['word']
        else:
            if current_entity:
                clean_tokens.append(current_entity)
            current_entity = ""
            current_entity_type = ""

    if current_entity and current_entity_type != 'person' and current_entity_type != 'pronoun':
        clean_tokens.append(current_entity)

    return clean_tokens


def embed_tokens(tokens_list: list) -> list:
    tokens_embeddings = []

    for token in tokens_list:
        try:
            embeddings = openai.Embedding.create(input=[token], model="text-embedding-ada-002")['data'][0]['embedding']
            embeddings = torch.tensor(embeddings).unsqueeze(0)

            tokens_embeddings.append(embeddings)
        except Exception as e:
            print("Error: {}, String: {}".format(e, token))
            continue

    return tokens_embeddings


def find_closest_nodes(entities_embeddings_list: [np.ndarray], graph_node_embeddings: np.ndarray, node_indices_list: list, prime_kg: nx.Graph) -> (list, list):
    """
    param entities_embeddings_list: a list of embedded entities (extracted from the query)
    param graph_node_embeddings: a matrix containing all node embeddings
    param node_indices_list: a list of primeKG's nodes' indices for quick lookup
    param prime_kg: PrimeKG as a nx.Graph
    return: a list of all closest nodes/ entities
    """

    closest_nodes_names = []
    closest_nodes_indices = []
    # index = faiss.IndexFlatL2(graph_node_embeddings.shape[0])
    # index.add(np.reshape(graph_node_embeddings.T, (graph_node_embeddings.T.shape[0], -1)))
    #
    # _, I = index.search(np.concatenate(entities_embeddings_list), 1)

    # Calculate the pairwise L2 norms
    # differences = np.concatenate(entities_embeddings_list)[:, np.newaxis, :] - graph_node_embeddings.numpy()[np.newaxis, :, :]
    distances = torch.norm(torch.cat(entities_embeddings_list).unsqueeze(1) - graph_node_embeddings.unsqueeze(0), p=2, dim=2)
    # l2_norms = np.sqrt(np.sum(differences ** 2, axis=-1))
    closest_embeddings_indices = torch.min(distances, dim=-1).indices

    for i in closest_embeddings_indices:
        closest_node_index = node_indices_list[i]
        closest_nodes_indices.append(closest_node_index)

        closest_node_name = prime_kg.nodes[closest_node_index]['name']
        closest_node_name = closest_node_name.replace(' ', '_')
        closest_nodes_names.append(closest_node_name)

    return closest_nodes_names, closest_nodes_indices
