import openai
import torch
import warnings

import networkx as nx
import numpy as np

from typing import List, Tuple

from config import OPENAI_API_KEY
from transformers import pipeline

openai.api_key = OPENAI_API_KEY

# Initialize the token classifier and embedding model
token_classifier = pipeline("token-classification", model="ukkendane/bert-medical-ner")
# Filter out transformers warning
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


def medical_ner(query_list: List[str], knowledge_graph_embeddings: np.ndarray, node_indices_list: list, prime_kg: nx.Graph) -> Tuple[List, List, List]:
    """
    Perform NER on medical queries
    Args:
        query_list: a list of string, containing all queries that necessitate ner
        knowledge_graph_embeddings: an embeddings tensor of the knowledge graph used for knowledge extraction
        node_indices_list: list of all the nodes indices from the kg for efficient mapping
        prime_kg: the knowledge graph used for knowledge extraction

    Returns:
        closest_entities: a list of the extracted string entities
        closest_entities_indices: the corresponding graph indices to closest_entities
        num_entities_list: a list of the number of extracted entities per entity in the original query_list

    """
    tokens = token_classifier(query_list)
    clean_tokens_list = clean_tokens(tokens)

    # Flatten tokens list before embedding them, store the length of each sublist for reconstruction
    flattened_tokens = [token for tokens_sublist in clean_tokens_list for token in tokens_sublist]
    num_entities_list = [len(sublist) for sublist in clean_tokens_list]

    if len(flattened_tokens) == 0:
        return [], [], []

    # Embedd tokens
    entities_embeddings_list = embed_tokens(flattened_tokens)

    if len(entities_embeddings_list) == 0:
        return [], [], []

    # Find the closest kg nodes
    closest_entities, closest_entities_indices = find_closest_nodes(entities_embeddings_list, knowledge_graph_embeddings, node_indices_list, prime_kg)

    return closest_entities, closest_entities_indices, num_entities_list


def clean_tokens(tokens_list: List[List[str]]) -> list:
    """
    A domain specific, model specific post-ner-processing method
    Args:
        tokens_list: the list of the returned tokens by the model utilized for ner

    Returns: the post-processed tokens

    """
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


def find_closest_nodes(entities_embeddings_list: List[torch.Tensor], graph_node_embeddings: torch.Tensor, node_indices_list: list, prime_kg: nx.Graph, threshold: float = 0.9) -> Tuple[list, list]:
    """

    Args:
        entities_embeddings_list: a list of embedded entities (extracted from the query)
        graph_node_embeddings: a tensor containing all node embeddings of the knowledge graph used for knowledge extraction
        node_indices_list: a list of all the nodes indices from the kg for efficient mapping
        prime_kg: the knowledge graph used for knowledge extraction
        threshold: optional. a similarity threshold for the distance computation

    Returns:
        closest_nodes_names: a list of strings of the found closest nodes content
        closest_nodes_indices: the corresponding graph indices to closest_nodes_names

    """

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert numpy arrays to torch tensors and move them to GPU
    entities_embeddings_tensor = torch.cat(entities_embeddings_list).to(device)
    graph_node_embeddings_tensor = graph_node_embeddings.to(device)

    # Calculate pairwise L2 norms
    distances = torch.norm(entities_embeddings_tensor.unsqueeze(1) - graph_node_embeddings_tensor.unsqueeze(0), p=2, dim=2)
    min_distances, closest_embeddings_indices = torch.min(distances, dim=-1)

    closest_nodes_names = []
    closest_nodes_indices = []

    for i in range(len(closest_embeddings_indices)):
        closest_node_index = node_indices_list[closest_embeddings_indices[i].item()]

        if min_distances[i].item() <= threshold:
            closest_nodes_indices.append(closest_node_index)
            closest_node_name = prime_kg.nodes[closest_node_index]['name'].replace(' ', '_')
            closest_nodes_names.append(closest_node_name)

    return closest_nodes_names, closest_nodes_indices
