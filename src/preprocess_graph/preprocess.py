import os
import argparse
import pickle
import torch

import pandas as pd
from datasets import load_dataset

from config import ROOT_DIR
from src.preprocess_graph.utils import node_types, relation_types
from src.preprocess_graph.build_subgraph import extract_knowledge_from_kg, create_graph_from_data, convert_nx_to_hetero_data, build_trie_from_kg
from src.preprocess_graph.medical_ner import medical_ner

parser = argparse.ArgumentParser(description='Preprocess KG on PrimeKG + Medmcqa')

parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--prime_kg_embeddings_dataset', type=str, default='KnowledgeExtraction/data/primeKG_embeddings.pt', help='primeKG embeddings pt path')
parser.add_argument('--node_embedding_to_name', type=str, default='datasets/node_embedding_to_name_dict.pkl', help='node embeddings to name dict path')
parser.add_argument('--trie_path', type=str, default='KnowledgeExtraction/data/trie.pickle', help='knowledge_graph_trie path, set to None if no trie is available')
parser.add_argument('--qa_dataset_name', type=str, default='medmcqa', help='Name of dataset to download using datasets')

args = parser.parse_args()

if __name__ == "__main__":
    knowledge_graph = pickle.load(open(os.path.join(ROOT_DIR, args.prime_kg_dataset), 'rb'))
    node_embeddings = torch.load(os.path.join(ROOT_DIR, args.prime_kg_embeddings_dataset))
    # node_embedding_to_name_dict = pickle.load(open(os.path.join(ROOT_DIR, args.node_embedding_to_name), 'rb'))

    dataset = load_dataset(args.qa_dataset_name)
    medmcqa_df = pd.DataFrame(dataset['validation'])

    if args.trie_path is None:
        trie = build_trie_from_kg(knowledge_graph, save_path=os.path.join(ROOT_DIR, 'KnowledgeExtraction/data'))
    else:
        trie = pickle.load(open(os.path.join(ROOT_DIR, args.trie_path), 'rb'))

    # iterate over medmcqa_df and add the questions to the graph
    for i, row in medmcqa_df.iterrows():
        original_question_series = row.drop(['id', 'cop', 'exp'])
        question = row['question']
        answer_choices = [row['opa'], row['opb'], row['opc'], row['opd']]

        question_entities_list = medical_ner(question, node_embeddings, node_embedding_to_name_dict)

        answer_entities_dict = {}
        answer_entities_list = []

        for answer_choice in ['opa', 'opb', 'opc', 'opd']:
            answer_entities = medical_ner(row[answer_choice], node_embeddings, node_embedding_to_name_dict)
            answer_entities_dict[answer_choice] = answer_entities
            answer_entities_list.append(answer_entities)

        if len(question_entities_list + answer_entities_list) == 0:
            continue

        raw_context_graph = extract_knowledge_from_kg(question, answer_choices, trie, knowledge_graph, question_entities_list, answer_entities_dict)

        processed_context_graph = create_graph_from_data(raw_context_graph)
        print('test')

    hetero_data = convert_nx_to_hetero_data(subgraph, node_types, relation_types, meta_relations_dict)
