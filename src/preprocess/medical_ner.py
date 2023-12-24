import networkx as nx
import numpy as np
import openai
import torch
from config import OPENAI_API_KEY
from transformers import pipeline

openai.api_key = OPENAI_API_KEY

# Initialize the token classifier and embedding model
token_classifier = pipeline("token-classification", model="ukkendane/bert-medical-ner")


def medical_ner(query_list: [str], knowledge_graph_embeddings: np.ndarray, node_indices_list: list, prime_kg: nx.Graph) -> (list, list, list):
    tokens = token_classifier(query_list)
    clean_tokens_list = clean_tokens(tokens)

    # Flatten tokens list before embedding them, store the length of each sublist for reconstruction
    flattened_tokens = [token for tokens_sublist in clean_tokens_list for token in tokens_sublist]
    num_entities_list = [len(sublist) for sublist in clean_tokens_list]

    # Embedd tokens
    entities_embeddings_list = embed_tokens(flattened_tokens)

    if len(entities_embeddings_list) == 0:
        return [], []

    # Find the closest kg nodes
    closest_entities, closest_entities_indices = find_closest_nodes(entities_embeddings_list, knowledge_graph_embeddings, node_indices_list, prime_kg)

    return closest_entities, closest_entities_indices, num_entities_list


def clean_tokens(tokens_list: [[str]]) -> list:
    clean_tokens_list = []

    for tokens in tokens_list:
        current_entity, current_entity_type = "", ""
        cleaned_tokens = []
        for token in tokens:
            if token['entity'].startswith('B_'):
                if current_entity and current_entity_type != 'person' and current_entity_type != 'pronoun':
                    cleaned_tokens.append(current_entity)
                current_entity = token['word']
                current_entity_type = token['entity'][2:]
            elif token['entity'].startswith('I_') and '##' in token['word']:
                current_entity += token['word'].replace('##', '')
            elif token['entity'].startswith('I_'):
                current_entity += " " + token['word']
            else:
                if current_entity:
                    cleaned_tokens.append(current_entity)
                current_entity = ""
                current_entity_type = ""

        if current_entity and current_entity_type != 'person' and current_entity_type != 'pronoun':
            cleaned_tokens.append(current_entity)

        clean_tokens_list.append(cleaned_tokens)

    return clean_tokens_list


def embed_tokens(tokens_list: list) -> list:
    try:
        embeddings = openai.Embedding.create(input=tokens_list, model="text-embedding-ada-002")['data']
        embeddings = [torch.tensor(item['embedding']).unsqueeze(0) for item in embeddings]

        return embeddings
    except Exception as e:
        print(f"Error in embedding tokens: {e}")
        return []


def find_closest_nodes(entities_embeddings_list: [np.ndarray], graph_node_embeddings: np.ndarray, node_indices_list: list, prime_kg: nx.Graph, threshold: float = 0.5) -> (list, list):
    """
    param entities_embeddings_list: a list of embedded entities (extracted from the query)
    param graph_node_embeddings: a matrix containing all node embeddings
    param node_indices_list: a list of primeKG's nodes' indices for quick lookup
    param prime_kg: PrimeKG as a nx.Graph
    return: a list of all closest nodes/ entities
    """

    closest_nodes_names = []
    closest_nodes_indices = []

    # Calculate the pairwise L2 norms
    distances = torch.norm(torch.cat(entities_embeddings_list).unsqueeze(1) - graph_node_embeddings.unsqueeze(0), p=2, dim=2)
    min_distances, closest_embeddings_indices = torch.min(distances, dim=-1)

    for i in range(len(closest_embeddings_indices)):
        closest_node_index = node_indices_list[closest_embeddings_indices[i]]

        if min_distances[i] <= threshold:
            closest_nodes_indices.append(closest_node_index)

            closest_node_name = prime_kg.nodes[closest_node_index]['name'].replace(' ', '_')
            closest_nodes_names.append(closest_node_name)

    return closest_nodes_names, closest_nodes_indices
