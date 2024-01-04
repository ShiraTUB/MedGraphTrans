import os
import argparse
import torch
import openai
import random
import pickle

import numpy as np
import networkx as nx
import pandas as pd

from typing import List
from config import ROOT_DIR, OPENAI_API_KEY
from KnowledgeExtraction.subgraph_builder import SubgraphBuilder
from src.utils import meta_relations_dict
from src.preprocess.medical_ner import medical_ner

openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser(description='Preprocess KG on PrimeKG + Medmcqa')

parser.add_argument('--prime_kg_dataset', type=str, default='datasets/prime_kg_nx_63960.pickle', help='PrimeKG pickle path')
parser.add_argument('--prime_kg_embeddings_dataset', type=str, default='datasets/prime_kg_embeddings_tensor_63960.pt', help='primeKG embeddings pt path')
parser.add_argument('--trie_path', type=str, default=None, help='knowledge_graph_trie path, set to None if no trie is available')
parser.add_argument('--qa_dataset_name', type=str, default='medmcqa', help='Name of dataset to download using datasets')
parser.add_argument('--dataset_target_path', type=str, default='datasets/', help='Path of where to save the processed dataset')
parser.add_argument('--target_dataset', type=str, default='train', help='Use either train/validation/test dataset')

args = parser.parse_args()


def initiate_question_graph(
        graph: nx.Graph,
        question: str,
        answer_choices: List[str],
        correct_answer: int,
        question_entities_indices_list: list,
        answer_entities_dict: dict,
        prime_kg: nx.Graph,
        question_index: int
) -> nx.Graph:
    """

    Args:
        graph: an empty nx.Graph()
        question: the question str from MedMCQA
        answer_choices: the corresponding answer choices to question
        correct_answer: the correct answer according the MedMCQA
        question_entities_indices_list: a list of entities extracted from question
        answer_entities_dict: a dictionary of answer choices and their corresponding extracted entities
        prime_kg: the kg from which we extract knowledge
        question_index: the index of the question in MedMCQA (it's placement in the data-set)

    Returns: the generated nx.Graph

    """

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect all texts for batch embedding
    all_texts = [question] + answer_choices
    embeddings = openai.Embedding.create(input=all_texts, model="text-embedding-ada-002")['data']
    embeddings = [torch.tensor(item['embedding']).to(device) for item in embeddings]
    question_embeddings = embeddings[0]

    graph.add_node(question_index, embedding=question_embeddings, type="question", index=question_index, name=question)

    for question_entity_index in question_entities_indices_list:
        target_node = prime_kg.nodes[question_entity_index]
        graph.add_node(question_entity_index, **target_node)
        graph.add_edge(question_index, question_entity_index, relation=f"question_{target_node['type']}")

    for choice_index, answer_embeddings in enumerate(embeddings[1:]):
        answer_index = random.randint(10 ** 9, (10 ** 10) - 1)
        graph.add_node(answer_index, embedding=answer_embeddings, type="answer", index=answer_index, name=answer_choices[choice_index], answer_choice_index=choice_index)

        if choice_index == correct_answer:
            graph.add_edge(question_index, answer_index, relation="question_correct_answer")
        else:
            graph.add_edge(question_index, answer_index, relation="question_wrong_answer")

        for answer_entity_index in answer_entities_dict[answer_choices[choice_index]]:
            target_node = prime_kg.nodes[answer_entity_index]
            graph.add_node(answer_entity_index, **target_node)
            graph.add_edge(answer_index, answer_entity_index, relation=f"answer_{target_node['type']}")

    return graph


def embed_text(text):
    """please implement this function according to your domain and use-case"""
    try:
        embeddings = openai.Embedding.create(input=[text], model="text-embedding-ada-002")['data'][0]['embedding']
        open_ai_embedding = np.reshape(np.asarray(embeddings), (-1, np.asarray(embeddings).shape[0]))
        return open_ai_embedding

    except Exception as e:
        print("Error: {}, String: {}".format(e, text))


if __name__ == "__main__":

    # load KG, node embeddings a trie and a dataset
    kg_path = os.path.join(ROOT_DIR, args.prime_kg_dataset)
    embeddings_path = os.path.join(ROOT_DIR, args.prime_kg_embeddings_dataset)
    trie_path = None if args.trie_path is None else os.path.join(ROOT_DIR, args.trie_path)
    trie_save_path = None if trie_path is not None else os.path.join(ROOT_DIR, args.dataset_target_path)

    subgraph_builder = SubgraphBuilder(kg_name_or_path=kg_path,
                                       kg_embeddings_path=embeddings_path,
                                       dataset_name_or_path=args.qa_dataset_name,
                                       meta_relation_types_dict=meta_relations_dict,
                                       embedding_method=embed_text,
                                       trie_path=trie_path,
                                       )

    # Save a list of all nodes indices for efficient mapping later on
    node_indices_list = [data['index'] for _, data in subgraph_builder.kg.nodes(data=True)]

    for purpose in ['train', 'validation', 'test']:

        args.target_dataset = purpose

        medmcqa_df = pd.DataFrame(subgraph_builder.dataset[args.target_dataset])

        question_uid_to_row_id_map = {}
        # iterate over medmcqa_df and create a graph per question

        for i, row in medmcqa_df.iterrows():

            subgraph_builder.nx_subgraph = nx.Graph()
            original_question_series = row.drop(['id', 'cop', 'exp'])
            question = row['question']
            answer_choices = [row['opa'], row['opb'], row['opc'], row['opd']]
            correct_answer = row['cop']
            row_id = row['id']

            entities_list, entities_indices_list, num_entities_list = medical_ner([question] + answer_choices, subgraph_builder.node_embeddings, node_indices_list, subgraph_builder.kg)

            if len(entities_list) == 0:
                continue

            # reconstruct answer indices for initiating the question graph:
            start = num_entities_list[0]
            end = start + num_entities_list[1]
            index = 2
            answer_entities_dict = {}

            for choice in answer_choices:
                answer_entities_dict[choice] = entities_indices_list[start:end]
                start = end
                end += num_entities_list[min(index, len(num_entities_list) - 1)]
                index += 1

            # Initiate question graph
            subgraph_builder.nx_subgraph = initiate_question_graph(subgraph_builder.nx_subgraph, question, answer_choices, correct_answer, entities_indices_list[:num_entities_list[0]], answer_entities_dict, subgraph_builder.kg,
                                                                   question_index=int(i))

            # Extract knowledge
            extracted_edges, extracted_edge_indices = subgraph_builder.extract_knowledge_from_kg(question, hops=2, neighbors_per_hop=10, entities_list=entities_list)

            if extracted_edge_indices is not None:
                subgraph_builder.expand_graph_with_knowledge(extracted_edge_indices)

            pickle.dump(subgraph_builder.nx_subgraph, open(os.path.join(ROOT_DIR, args.dataset_target_path, args.target_dataset, f'graph_{i}.pickle'), 'wb'))
            print('processed {} out of {} rows'.format(i, len(medmcqa_df)))
