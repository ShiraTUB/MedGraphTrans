import os
import openai
import argparse
import torch
import pickle

from tqdm import tqdm
import pandas as pd
import networkx as nx
from config import OPENAI_API_KEY, ROOT_DIR

openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser(description='Preprocess PrimeKG')
parser.add_argument('--prime_kg_dataset', type=str, default='datasets/kg.csv', help='PrimeKG csv path')
args = parser.parse_args()


class GraphBuilder:
    def __init__(self,
                 prime_kg_path: str,
                 drug_features_path: str,
                 disease_features_path: str,
                 node_types: list,
                 edge_types: list,
                 save_path: str = None,
                 filter_edge_or_node_types: list = None):

        self.node_types = node_types
        self.edge_types = edge_types
        self.type_index_dict = {ntype: [] for ntype in self.node_types + self.edge_types}

        self.prime_kg_df = pd.read_csv(os.path.join(ROOT_DIR, prime_kg_path))

        if filter_edge_or_node_types is not None:
            self.create_sub_df(filter_edge_or_node_types)

        self.drug_features_df = pd.read_csv(os.path.join(ROOT_DIR, drug_features_path), low_memory=False)
        self.disease_features_df = pd.read_csv(os.path.join(ROOT_DIR, disease_features_path), low_memory=False)

        self.embeddings_tensor = None
        self.embeddings_list = []
        self.nx_graph = nx.Graph()
        self.model = "text-embedding-ada-002"
        self.save_path = save_path

        self.generate_nx_graph()
        self.validate_edges()

        if save_path is not None:
            # save graph object to file
            file_name = f"prime_gk_nx_{len(self.nx_graph.nodes())}"
            pickle.dump(self.nx_graph, open(os.path.join(ROOT_DIR, save_path, file_name), 'wb'))

    def create_sub_df(self, types: list, filter_nodes: bool = False):
        df = self.prime_kg_df

        if filter_nodes:
            sub_df = df[df['x_type'].isin(types) and df['y_type'].isin(types)]
        else:
            sub_df = df[df['relation'].isin(types)]

        sub_df.to_csv(os.path.join(ROOT_DIR, 'datasets/prime_kg_{}.csv'.format(", ".join(types))), index=False)
        self.prime_kg_df = sub_df

    def generate_nx_graph(self):

        self.nx_graph = nx.from_pandas_edgelist(
            self.prime_kg_df,
            source="x_index",
            target="y_index",
            edge_attr='relation',
            create_using=nx.Graph()
        )

        for node in tqdm(list(self.nx_graph.nodes())):
            x_sub_df = self.prime_kg_df.query(f'x_index == {node}')

            # get all edge_types and their respective edges
            if not x_sub_df.empty:
                for y_index in list(x_sub_df['y_index']):
                    rel_type = self.nx_graph.get_edge_data(node, y_index, 0)['relation']
                    self.type_index_dict[rel_type].append((node, y_index))

            # create an attributes dictionary for the relevant node
            x_sub_df = x_sub_df[['x_index', 'x_type', 'x_name', 'x_source']].drop_duplicates().rename(
                columns={'x_index': 'index', 'x_type': 'type', 'x_name': 'name', 'x_source': 'source'})

            y_sub_df = self.prime_kg_df.query(f'y_index == {node}')
            y_sub_df = y_sub_df[['y_index', 'y_type', 'y_name', 'y_source']].drop_duplicates().rename(
                columns={'y_index': 'index', 'y_type': 'type', 'y_name': 'name', 'y_source': 'source'})

            node_attributes_dict = pd.concat([x_sub_df, y_sub_df]).drop_duplicates().iloc[0].to_dict()

            # Extract disease and drug features
            features_sub_df = None

            if node_attributes_dict.get('type') == 'disease':
                features_sub_df = self.disease_features_df.query(f'node_index == {node}')
                features_sub_df = features_sub_df[
                    ['mondo_definition', 'umls_description', 'orphanet_definition',
                     'orphanet_prevalence']].drop_duplicates()
                features_sub_df = combine_rows(features_sub_df)

            elif node_attributes_dict.get('type') == 'drug':

                features_sub_df = self.drug_features_df.query(f'node_index == {node}')
                features_sub_df = features_sub_df[['description', 'indication']].drop_duplicates()
                features_sub_df = combine_rows(features_sub_df)

            raw_node_data = f"'name': '{node_attributes_dict.get('name')}', " + f"'type': '{node_attributes_dict.get('type')}'"

            if features_sub_df is not None:
                raw_node_data += ', ' + ', '.join(
                    f"'{col_name}': '{cell_value}'" for col_name, cell_value in features_sub_df.iloc[0].items())

            raw_node_data = raw_node_data.replace("\n", " ")
            self.nx_graph.nodes[node]['raw_data'] = raw_node_data

            self.nx_graph.nodes[node]['type'] = node_attributes_dict.get('type')
            self.nx_graph.nodes[node]['index'] = node_attributes_dict.get('index')
            self.nx_graph.nodes[node]['name'] = node_attributes_dict.get('name')
            self.nx_graph.nodes[node]['source'] = node_attributes_dict.get('source')
            self.type_index_dict[node_attributes_dict.get('type')].append(node_attributes_dict.get('index'))

            if len(list(nx.get_node_attributes(self.nx_graph, 'raw_data').values())) >= 1000:
                break

        # Embedd Nodes in batches according to the number of tokens per node to minimize the number of api calls
        max_input_tokes = 8191
        nodes_data = list(nx.get_node_attributes(self.nx_graph, 'raw_data').values())
        num_tokens_per_node = [(len(node_data.split()) * 3) for node_data in nodes_data]
        nodes_index = list(nx.get_node_attributes(self.nx_graph, 'index').values())
        nodes_type = list(nx.get_node_attributes(self.nx_graph, 'type').values())

        prompts_list = []
        start = 0
        for i, (node_data, node_index, node_type) in enumerate(list(zip(nodes_data, nodes_index, nodes_type))):
            end = i
            number_of_tokens = len(node_data.split()) * 3
            if number_of_tokens > max_input_tokes:
                prompt_string = "".join(node_data.split()[:2600])
            else:
                prompt_string = node_data
            prompts_list.append(prompt_string)

            if end - start == 10:
                try:
                    response = openai.Embedding.create(input=prompts_list, model=self.model)
                    embeddings_list = [torch.tensor(item['embedding']).unsqueeze(1) for item in response['data']]

                    for idx, node_idx in enumerate(nodes_index[start:end]):
                        embedding = embeddings_list[idx]
                        self.nx_graph.nodes[node_idx]['embedding'] = embedding
                        self.embeddings_list.append(embedding)

                    start = end

                except Exception as e:
                    print("Error: {}, Indices: {}-{}".format(e, start, end))

        self.embeddings_tensor = torch.cat(self.embeddings_list, dim=1)

        if self.save_path is not None:
            file_name = f'prime_kg_embeddings_tensor_{self.embeddings_tensor.size(0)}'
            pickle.dump(self.embeddings_tensor, open(os.path.join(ROOT_DIR, self.save_path, file_name), 'wb'))

    def validate_edges(self):
        # Convert the DataFrame to a set of tuples for fast lookup
        original_edges = set(zip(self.prime_kg_df['x_index'], self.prime_kg_df['y_index']))

        # Initialize a list to store any edges that don't match
        mismatched_edges = []

        # Iterate over each edge in the graph
        for source, target in self.nx_graph.edges():
            # Convert edge to the same type as DataFrame values for comparison (if needed)
            source = type(self.prime_kg_df['x_index'].iloc[0])(source)
            target = type(self.prime_kg_df['y_index'].iloc[0])(target)

            # Check if the edge exists in the original data
            if (source, target) not in original_edges:
                mismatched_edges.append((source, target))

        # Report the results
        if mismatched_edges:
            print(f"Found {len(mismatched_edges)} mismatched edges.")
            # Optionally print the mismatched edges
            for edge in mismatched_edges:
                print(edge)
        else:
            print("All edges in the graph match the original dataset.")


def combine_rows(df: pd.DataFrame) -> pd.DataFrame:
    combined_row = {}
    for col in df.columns:

        combined_row[col] = df[col].first_valid_index()

        if combined_row[col] is not None:
            combined_row[col] = df[col].loc[combined_row[col]]

    combined_df = pd.DataFrame(combined_row, index=[0])
    return combined_df


node_types = ['gene/protein', 'drug', 'effect/phenotype', 'disease', 'biological_process', 'molecular_function',
              'cellular_component', 'exposure', 'pathway', 'anatomy']

edge_types = ["protein_protein", "drug_protein", "contraindication", "indication", "off_label_use", "drug_drug",
              "phenotype_protein", "phenotype_phenotype", "disease_phenotype_negative",
              "disease_phenotype_positive", "disease_protein", "disease_disease", "drug_effect",
              "bioprocess_bioprocess", "molfunc_molfunc", "cellcomp_cellcomp", "molfunc_protein",
              "cellcomp_protein", "bioprocess_protein", "exposure_protein", "exposure_disease", "exposure_exposure",
              "exposure_bioprocess", "exposure_molfunc", "exposure_cellcomp", "pathway_pathway", "pathway_protein",
              "anatomy_anatomy", "anatomy_protein_present", "anatomy_protein_absent"]

prime_kg_path = 'datasets/kg.csv'
drug_features_path = 'datasets/drug_features.csv'
disease_features_path = 'datasets/disease_features.csv'

filter_edge_types = ["indication", "drug_drug", "phenotype_phenotype", "disease_phenotype_positive", "disease_disease", "drug_effect", ]

gb = GraphBuilder(prime_kg_path, drug_features_path, disease_features_path, node_types=node_types, edge_types=edge_types, filter_edge_or_node_types=filter_edge_types)
