import pandas as pd
import networkx as nx
import tqdm
import torch
import numpy as np
from datetime import date
import openai

from torch_geometric.data import HeteroData
from src.utils import meta_relations_dict_complete
from config import OPENAI_API_KEY


class GraphBuilder:
    def __init__(self, prime_kg_df, drug_features_path, disease_features_path, output_dim=1536, embeddings_path=None,
                 graph=None):
        self.output_dim = output_dim
        self.type_index_dict = {ntype: [] for ntype in self.node_types + self.edge_types}
        self.nx_graph = nx.MultiDiGraph()
        self.hetero_data = HeteroData()
        self.all_reps_emb = []
        self.model = "text-embedding-ada-002"

        if graph:
            self.nx_graph = graph
            ntypes = nx.get_node_attributes(graph, 'type')
            etypes = nx.get_edge_attributes(graph, 'relation')

            for ntype in self.node_types:
                self.type_index_dict[ntype] = [node for node in graph.nodes() if ntypes[node] == ntype]

            for etype in self.edge_types:
                self.type_index_dict[etype] = [edge for edge in graph.edges() if etypes[(edge[0], edge[1], 0)] == etype]
        else:
            if prime_kg_df is not None:
                self.generate_nx_graph(prime_kg_df, drug_features_path, disease_features_path, embeddings_path)
                self.validate_edges()

                # save graph object to file
                # today = date.today()
                # today_str = today.strftime("%Y-%m-%d")
                # pickle.dump(self.nx_graph, open(f"../dd_datasets/nx_graph_{today_str}.pickle", 'wb'))

        self.convert_nx_to_hetero()

    node_types = ['gene/protein', 'drug', 'effect/phenotype', 'disease', 'biological_process', 'molecular_function',
                  'cellular_component', 'exposure', 'pathway', 'anatomy']

    edge_types = ["protein_protein", "drug_protein", "contraindication", "indication", "off_label_use", "drug_drug",
                  "phenotype_protein", "phenotype_phenotype", "disease_phenotype_negative",
                  "disease_phenotype_positive", "disease_protein", "disease_disease", "drug_effect",
                  "bioprocess_bioprocess", "molfunc_molfunc", "cellcomp_cellcomp", "molfunc_protein",
                  "cellcomp_protein", "bioprocess_protein", "exposure_protein", "exposure_disease", "exposure_exposure",
                  "exposure_bioprocess", "exposure_molfunc", "exposure_cellcomp", "pathway_pathway", "pathway_protein",
                  "anatomy_anatomy", "anatomy_protein_present", "anatomy_protein_absent"]

    def generate_nx_graph(self, prime_kg_df, drug_features_path, disease_features_path, embeddings_path=None):
        # prime_kg_df = pd.read_csv(prime_kg_df, low_memory=False)
        drug_features_df = pd.read_csv(drug_features_path, low_memory=False)
        disease_features_df = pd.read_csv(disease_features_path, low_memory=False)
        embeddings_df = None

        if embeddings_path is not None:
            embeddings_df = pd.read_csv(embeddings_path)

        self.nx_graph = nx.from_pandas_edgelist(
            prime_kg_df,
            source="x_index",
            target="y_index",
            edge_attr='relation',
            create_using=nx.MultiDiGraph()
        )

        embeddings_dict = {}

        if embeddings_df is not None:
            embeddings_csv_to_dict(embeddings_df, embeddings_dict)

        for node in self.nx_graph.nodes():
            x_sub_df = prime_kg_df.query(f'x_index == {node}')

            # get all edge_types and their respective edges
            if not x_sub_df.empty:
                for y_index in list(x_sub_df['y_index']):
                    rel_type = self.nx_graph.get_edge_data(node, y_index, 0)['relation']
                    self.type_index_dict[rel_type].append((node, y_index))

            # create an attributes dictionary for the relevant node
            x_sub_df = x_sub_df[['x_index', 'x_type', 'x_name', 'x_source']].drop_duplicates().rename(
                columns={'x_index': 'index', 'x_type': 'type', 'x_name': 'name', 'x_source': 'source'})

            y_sub_df = prime_kg_df.query(f'y_index == {node}')
            y_sub_df = y_sub_df[['y_index', 'y_type', 'y_name', 'y_source']].drop_duplicates().rename(
                columns={'y_index': 'index', 'y_type': 'type', 'y_name': 'name', 'y_source': 'source'})

            node_attributes_dict = pd.concat([x_sub_df, y_sub_df]).drop_duplicates().iloc[0].to_dict()
            embeddings = None

            if embeddings_df is not None:
                if f"{node}" in embeddings_df.keys():
                    embeddings = torch.reshape(torch.from_numpy(embeddings_dict[node]), (-1, 1))
            else:
                features_sub_df = None

                if node_attributes_dict.get('type') == 'disease':
                    features_sub_df = disease_features_df.query(f'node_index == {node}')
                    features_sub_df = features_sub_df[
                        ['mondo_definition', 'umls_description', 'orphanet_definition',
                         'orphanet_prevalence']].drop_duplicates()
                    features_sub_df = combine_rows(features_sub_df)

                elif node_attributes_dict.get('type') == 'drug':

                    features_sub_df = drug_features_df.query(f'node_index == {node}')
                    features_sub_df = features_sub_df[['description', 'indication']].drop_duplicates()
                    features_sub_df = combine_rows(features_sub_df)

                raw_node_data = f"'name': '{node_attributes_dict.get('name')}', " + f"'type': '{node_attributes_dict.get('type')}'"

                if features_sub_df is not None:
                    raw_node_data += ', ' + ', '.join(
                        f"'{col_name}': '{cell_value}'" for col_name, cell_value in features_sub_df.iloc[0].items())

                raw_node_data = raw_node_data.replace("\n", " ")

                try:
                    embeddings = openai.Embedding.create(input=[raw_node_data], model=self.model)['data'][0][
                        'embedding']
                    embeddings_dict[node] = np.asarray(embeddings)
                except Exception as e:
                    print("Error: {}, Index: {}".format(e, node_attributes_dict.get('index')))
                    continue

                self.nx_graph.nodes[node]['raw_data'] = raw_node_data

            if embeddings is not None:
                self.all_reps_emb.append(embeddings)
                self.nx_graph.nodes[node]['embedding'] = embeddings

            self.nx_graph.nodes[node]['type'] = node_attributes_dict.get('type')
            self.nx_graph.nodes[node]['index'] = node_attributes_dict.get('index')
            self.nx_graph.nodes[node]['name'] = node_attributes_dict.get('name')
            self.nx_graph.nodes[node]['source'] = node_attributes_dict.get('source')
            self.type_index_dict[node_attributes_dict.get('type')].append(node_attributes_dict.get('index'))

        self.all_reps_emb = np.concatenate(self.all_reps_emb, axis=1)

        if embeddings_path is None:
            embeddings_df = pd.DataFrame(embeddings_dict)
            today = date.today()
            today_str = today.strftime("%Y-%m-%d")
            embeddings_df.to_csv(f"../datasets/embeddings_{today_str}.csv", index=False)

    # def convert_nx_to_hetero(self):
    #     for node_type in self.node_types:
    #         embeddings = []
    #         node_type_indices = self.type_index_dict[node_type]
    #
    #         for node_index in node_type_indices:
    #             embeddings.append(self.nx_graph.nodes[node_index]['embedding'])
    #
    #         if len(embeddings) > 0:
    #             self.hetero_data[node_type].x = torch.stack(embeddings, dim=0).type("torch.FloatTensor")
    #
    #             # TODO: find the right masking strategy
    #             if node_type == 'disease':
    #                 train_mask, val_mask, test_mask = self.tvt_split(len(embeddings))
    #                 self.hetero_data[node_type].update(
    #                     dict(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask))
    #
    #     for edge_type in self.edge_types:
    #         meta_rel_type = meta_relations_dict_complete[edge_type]
    #
    #         source_indices = list([s for s, _ in self.type_index_dict[edge_type]])
    #         target_indices = list([t for _, t in self.type_index_dict[edge_type]])
    #
    #         edge_index_feats = [torch.tensor(source_indices), torch.tensor(target_indices)]
    #
    #         if len(edge_index_feats[0]) != len(edge_index_feats[1]):
    #             continue
    #         elif len(edge_index_feats[0]) == 0:
    #             continue
    #
    #         self.hetero_data[meta_rel_type[0], meta_rel_type[1], meta_rel_type[2]].edge_index = torch.stack(
    #             edge_index_feats, dim=0)

    def validate_edges(self):
        # Convert the DataFrame to a set of tuples for fast lookup
        original_edges = set(zip(prime_kg_df['x_index'], prime_kg_df['y_index']))

        # Initialize a list to store any edges that don't match
        mismatched_edges = []

        # Iterate over each edge in the graph
        for source, target in self.nx_graph.edges():
            # Convert edge to the same type as DataFrame values for comparison (if needed)
            source = type(prime_kg_df['x_index'].iloc[0])(source)
            target = type(prime_kg_df['y_index'].iloc[0])(target)

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


def embeddings_csv_to_dict(embeddings_df, embeddings_dict):
    for column in embeddings_df.columns:
        embeddings_dict[int(column)] = embeddings_df[column].values


def combine_rows(df):
    combined_row = {}
    for col in df.columns:

        combined_row[col] = df[col].first_valid_index()

        if combined_row[col] is not None:
            combined_row[col] = df[col].loc[combined_row[col]]

    combined_df = pd.DataFrame(combined_row, index=[0])
    return combined_df


prime_kg_df = pd.read_csv('../../datasets/kg.csv')
random_toy_prime_kg_df = prime_kg_df.sample(n=200)

drug_features_path = '../../datasets/drug_features.csv'
disease_features_path = '../../datasets/disease_features.csv'

gb = GraphBuilder(random_toy_prime_kg_df, drug_features_path, disease_features_path)
