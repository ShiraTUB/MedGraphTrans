import os
import argparse
import pickle
import torch

import networkx as nx
import pandas as pd
from datasets import load_dataset

from config import ROOT_DIR
from KnowledgeExtraction.subgraph_builder import SubgraphBuilder
from src.preprocess_graph.build_subgraph import initiate_question_graph
from src.utils import meta_relations_dict, embed_text
from src.preprocess_graph.medical_ner import medical_ner

parser = argparse.ArgumentParser(description='Preprocess KG on PrimeKG + Medmcqa')

parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--prime_kg_embeddings_dataset', type=str, default='datasets/primeKG_embeddings_tensor.pt', help='primeKG embeddings pt path')
parser.add_argument('--trie_path', type=str, default='datasets/trie.pickle', help='knowledge_graph_trie path, set to None if no trie is available')
parser.add_argument('--qa_dataset_name', type=str, default='medmcqa', help='Name of dataset to download using datasets')
parser.add_argument('--dataset_target_path', type=str, default='datasets/raw_graph_dataset', help='Path of where to save the processed dataset')
parser.add_argument('--target_dataset', type=str, default='train', help='Use either train/validation/test dataset')

args = parser.parse_args()

if __name__ == "__main__":

    # load KG, node embeddings a trie and a dataset
    kg_path = os.path.join(ROOT_DIR, args.prime_kg_dataset)
    embeddings_path = os.path.join(ROOT_DIR, args.prime_kg_embeddings_dataset)
    trie_path = None if args.trie_path is None else os.path.join(ROOT_DIR, args.trie_path)
    trie_save_path = None if trie_path is not None else os.path.join(ROOT_DIR, args.trie_save_path)

    subgraph_builder = SubgraphBuilder(kg_name_or_path=kg_path,
                                       kg_embeddings_path=embeddings_path,
                                       dataset_name_or_path=args.qa_dataset_name,
                                       meta_relation_types_dict=meta_relations_dict,
                                       embedding_method=embed_text,
                                       trie_path=trie_path,
                                       )

    node_indices_list = [data['index'] for _, data in subgraph_builder.kg.nodes(data=True)]

    for purpose in ['train', 'validation', 'test']:

        args.target_dataset = purpose

        medmcqa_df = pd.DataFrame(subgraph_builder.dataset[args.target_dataset])[:500]

        # iterate over medmcqa_df and add the questions to the graph
        for i, row in medmcqa_df.iterrows():

            subgraph_builder.nx_subgraph = nx.Graph()
            original_question_series = row.drop(['id', 'cop', 'exp'])
            question = row['question']
            answer_choices = [row['opa'], row['opb'], row['opc'], row['opd']]
            correct_answer = row['cop']

            question_entities_list, question_entities_indices_list = medical_ner(question, subgraph_builder.node_embeddings, node_indices_list, subgraph_builder.kg)

            answer_entities_dict = {}
            answer_entities_list = []

            for answer_choice in answer_choices:
                answer_entities, answer_entities_indices_list = medical_ner(answer_choice, subgraph_builder.node_embeddings, node_indices_list, subgraph_builder.kg)
                answer_entities_dict[answer_choice] = answer_entities_indices_list
                answer_entities_list.append(answer_entities)

            # Flatten answer_entities_list
            answers_entities_list = [entity for entities in answer_entities_list for entity in entities]

            if len(question_entities_list + answer_entities_list) == 0:
                continue

            subgraph_builder.nx_subgraph = initiate_question_graph(subgraph_builder.nx_subgraph, question, answer_choices, correct_answer, question_entities_indices_list, answer_entities_dict, subgraph_builder.kg)

            extracted_edges, extracted_edge_indices = subgraph_builder.extract_knowledge_from_kg(question, hops=2, neighbors_per_hop=10, entities_list=question_entities_list+answers_entities_list)

            if extracted_edge_indices is not None:
                subgraph_builder.expand_graph_with_knowledge(extracted_edge_indices)

            pickle.dump(subgraph_builder.nx_subgraph, open(os.path.join(ROOT_DIR, args.dataset_target_path, args.target_dataset, f'graph_{i}.pickle'), 'wb'))
            print('processed {} out of {} rows'.format(i, len(medmcqa_df)))

