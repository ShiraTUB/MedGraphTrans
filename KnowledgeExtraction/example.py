import os
import pickle

import pandas as pd
import torch
from datasets import load_dataset

from src.preprocess_graph.build_subgraph import build_trie_from_kg, extract_knowledge_from_kg, convert_nx_to_hetero_data

# load KG, node embeddings a trie and a dataset
root_dir = os.path.dirname(os.path.abspath(__file__))

knowledge_graph_path = os.path.join(root_dir, 'data/primeKG_nx_medium.pickle')
knowledge_graph = pickle.load(open(knowledge_graph_path, 'rb'))

embeddings_path = os.path.join(root_dir, 'data/primeKG_embeddings.pt')
node_embeddings = torch.load(embeddings_path)

trie_path = '../datasets/trie.pickle'  # set to None if no trie is available
# trie_path = None

if trie_path is None:
    trie = build_trie_from_kg(knowledge_graph, save_path='./data')

else:
    trie_path = os.path.join(root_dir, trie_path)
    trie = pickle.load(open(trie_path, 'rb'))

dataset = load_dataset("medmcqa")
dataset_df = pd.DataFrame(dataset['train'])

query_example = dataset_df.iloc[0]['question']

# for each entry of the dataset, extract the most relevant subgraph out of the trie

# optional: if you kg/ dataset in not a dialog or a general NL dataset, domain specific ner might be necessary before
#  searching the trie. In this case medical ner was performed and these are the extracted entities:
entities = ['urethral_obstruction', 'benign_prostate_phyllodes_tumor', 'kidney_hypertrophy']

subgraph = extract_knowledge_from_kg(query_example, trie, knowledge_graph, entities)

print('The extracted subgraph:\n')
for u, v, data in subgraph.edges(data=True):
    print(f"source: {subgraph.nodes[u]['name']} - relation: {data['relation']} - target: {subgraph.nodes[v]['name']}")

"""The extracted subgraph can be then taken into the next stage of the pipeline --> '
      'build a dataset, train a model, be passed to a transformer ect."""
######################################################################################
"""Below is an example of how to convert the extracted nx graph into HeteroData
    please note that all of the variables are domain and kg specific."""

relation_types = ["indication", "phenotype_protein", "phenotype_phenotype", "disease_phenotype_positive",
                  "disease_protein", "disease_disease", "drug_effect"]

meta_relations_dict = {
    "indication": ('drug', 'indication', 'disease'),
    "phenotype_protein": ('effect/phenotype', 'phenotype_protein', 'gene/protein'),
    "phenotype_phenotype": ('effect/phenotype', 'phenotype_phenotype', 'effect/phenotype'),
    "disease_phenotype_positive": ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
    "disease_protein": ('disease', "disease_protein", 'gene/protein'),
    "disease_disease": ('disease', 'disease_disease', 'disease'),
    "drug_effect": ('drug', 'drug_effect', 'effect/phenotype')}

node_types = ['drug', 'disease', 'effect/phenotype', 'gene/protein']

hetero_data = convert_nx_to_hetero_data(subgraph, node_types, relation_types, meta_relations_dict)

print(hetero_data)
