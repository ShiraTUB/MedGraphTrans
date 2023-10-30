import os
import argparse
import pickle
import torch

import networkx as nx
import pandas as pd
from datasets import load_dataset

from config import ROOT_DIR
from src.preprocess_graph.build_subgraph import extract_knowledge_from_kg, build_trie_from_kg, initiate_question_graph, expand_graph_with_knowledge, convert_nx_to_hetero_data
from src.preprocess_graph.medical_ner import medical_ner

parser = argparse.ArgumentParser(description='Preprocess KG on PrimeKG + Medmcqa')

parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--prime_kg_embeddings_dataset', type=str, default='KnowledgeExtraction/data/primeKG_embeddings.pt', help='primeKG embeddings pt path')
parser.add_argument('--node_embedding_to_name', type=str, default='datasets/node_embedding_to_name_dict.pkl', help='node embeddings to name dict path')
parser.add_argument('--trie_path', type=str, default='KnowledgeExtraction/data/trie.pickle', help='knowledge_graph_trie path, set to None if no trie is available')
parser.add_argument('--qa_dataset_name', type=str, default='medmcqa', help='Name of dataset to download using datasets')
parser.add_argument('--dataset_target_path', type=str, default='datasets/graph_dataset', help='Path of where to save the processed dataset')
parser.add_argument('--target_dataset', type=str, default='train', help='Use either train/val/test dataset')

args = parser.parse_args()

relation_types = ["indication", "phenotype_protein", "phenotype_phenotype", "disease_phenotype_positive",
                  "disease_protein", "disease_disease", "drug_effect", "question_knowledge", "answer_knowledge", "question_answer"]

meta_relations_dict = {
    "indication": ('drug', 'indication', 'disease'),
    "phenotype_protein": ('effect/phenotype', 'phenotype_protein', 'gene/protein'),
    "phenotype_phenotype": ('effect/phenotype', 'phenotype_phenotype', 'effect/phenotype'),
    "disease_phenotype_positive": ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
    "disease_protein": ('disease', "disease_protein", 'gene/protein'),
    "disease_disease": ('disease', 'disease_disease', 'disease'),
    "drug_effect": ('drug', 'drug_effect', 'effect/phenotype'),
    "question_knowledge": ('question', 'question_knowledge', 'knowledge'),
    "answer_knowledge": ('answer', 'answer_knowledge', 'knowledge'),
    "question_answer": ('question', 'question_answer', 'answer'),
}

node_types = ['drug', 'disease', 'effect/phenotype', 'gene/protein', 'question', 'answer']

if __name__ == "__main__":
    prime_kg = pickle.load(open(os.path.join(ROOT_DIR, args.prime_kg_dataset), 'rb'))
    node_embeddings = torch.load(os.path.join(ROOT_DIR, args.prime_kg_embeddings_dataset))
    node_embedding_to_name_dict = pickle.load(open(os.path.join(ROOT_DIR, args.node_embedding_to_name), 'rb'))

    dataset = load_dataset(args.qa_dataset_name)
    medmcqa_df = pd.DataFrame(dataset[args.target_dataset])

    if args.trie_path is None:
        trie = build_trie_from_kg(prime_kg, save_path=os.path.join(ROOT_DIR, 'KnowledgeExtraction/data'))
    else:
        trie = pickle.load(open(os.path.join(ROOT_DIR, args.trie_path), 'rb'))

    medmcqa_df = medmcqa_df[:100]

    # iterate over medmcqa_df and add the questions to the graph
    for i, row in medmcqa_df.iterrows():
        graph = nx.Graph()

        original_question_series = row.drop(['id', 'cop', 'exp'])
        question = row['question']
        answer_choices = [row['opa'], row['opb'], row['opc'], row['opd']]

        question_entities_list, question_entities_indices_list = medical_ner(question, node_embeddings, node_embedding_to_name_dict)

        answer_entities_dict = {}
        answer_entities_list = []

        for answer_choice in answer_choices:
            answer_entities, answer_entities_indices_list = medical_ner(answer_choice, node_embeddings, node_embedding_to_name_dict)
            answer_entities_dict[answer_choice] = answer_entities_indices_list
            answer_entities_list.append(answer_entities)

        if len(question_entities_list + answer_entities_list) == 0:
            continue

        graph = initiate_question_graph(graph, question, answer_choices, question_entities_indices_list, answer_entities_dict, prime_kg)

        extracted_edges, extracted_edge_indices = extract_knowledge_from_kg(question, trie, question_entities_list, answer_entities_list)

        graph = expand_graph_with_knowledge(graph, extracted_edge_indices, prime_kg)

        hetero_data = convert_nx_to_hetero_data(graph, node_types, relation_types, meta_relations_dict)

        pickle.dump(hetero_data, open(os.path.join(ROOT_DIR, args.dataset_target_path, args.target_dataset, 'processed', f'part_{i}.pickle'), 'wb'))

        print('processed {} out of {} rows'.format(i, len(medmcqa_df)))
