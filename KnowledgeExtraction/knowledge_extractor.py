import numpy as np
import faiss
import openai
from gensim.parsing.preprocessing import STOPWORDS
from nltk import RegexpTokenizer
from KnowledgeExtraction.trie_structure import Trie

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class KnowledgeExtractor:
    def __init__(self, trie: Trie, text: str, entities=None):
        """
        self.trie: The trie from which knowledge is extracted
        self.text: The query/ sentence on which ner is performed
        self.data:Can be used later on for subgraph reconstruction. Depends on the use case
        self.model: The model used for embedding the context node
        self.stopwords: Used for extracting the entities from the input text if no entities list is given
        self.context_node_embedding: The embedding of the input text
        self.entities: The extracted entities of the input text
        self.knowledge_triplets_list: The extracted edges triplets from all hops
        self.knowledge_nodes_embedding = The corresponding to self.knowledge_triplets_list ->
            per triplet- the embeddings of it's target node are stored
        self.knowledge_indices: The corresponding node indices to the edges in knowledge_nodes_embedding (optional)
        """

        self.trie = trie
        self.text = text
        self.data = {}
        self.model = "text-embedding-ada-002"  # domain specific
        self.stopwords = STOPWORDS.union(set(['I', 'you', 'he', 'she', 'it', 'we', 'they']))
        self.context_node_embedding = None
        self.entities = None
        self.knowledge_triplets_list = None
        self.knowledge_nodes_embedding = None
        self.knowledge_indices = None
        self.set_context_node_embedding(self.model)
        self.set_entities(entities)
        self.init_data()

    def init_data(self):
        """make sure to make the necessary changes to your domain.
        In some cased this step can be skipped.

        Below is an example of the data initiation for creating the DiaTransNet datasets"""
        tail = np.char.array(self.entities)
        l = len(tail)
        head = np.char.array(self.text)
        head = np.resize(head, (l))

        relation = np.char.array("known")
        relation = np.resize(relation, (l))
        edges = np.char.array([head, tail, relation]).T
        self.data['msg_knowledge_edges'] = edges
        self.data['knowledge_nodes'] = tail

    def get_data(self, key):
        return self.data[key]

    def get_stopwords(self):
        return self.stopwords

    def set_context_node_embedding(self, model):
        """please implement this function according to your domain and use-case"""
        # self.context_node_embedding = self.model.encode([self.text])
        try:
            embeddings = openai.Embedding.create(input=[self.text], model=model)['data'][0]['embedding']
            self.context_node_embedding = np.reshape(np.asarray(embeddings), (-1, np.asarray(embeddings).shape[0]))

        except Exception as e:
            print("Error: {}, String: {}".format(e, self.text))

    def get_text_embedding(self):
        return self.context_node_embedding

    def set_entities(self, entities=None):
        if entities is None:
            tokenizer = RegexpTokenizer(r"\w+")
            text_tokenized = tokenizer.tokenize(self.text)
            basic_entities = [word for word in text_tokenized if not word.lower() in self.stopwords]
            self.entities = basic_entities
        else:
            self.entities = entities

    def get_entities(self):
        return self.entities

    def check_emb(self, emb):
        if type(emb) != np.ndarray:
            emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb

    def search_neighborhood(self, k):
        """
        :param k: number of nodes to be includes in the neighborhood
        :return: the extracted neighborhood and its corresponding indices (optional)
        """
        if self.knowledge_triplets_list is None:
            return None, None

        # extract the doublet trippels
        tuppel = np.char.array([self.knowledge_triplets_list[:, 0], self.knowledge_triplets_list[:, 1]]).T
        cleared_tuppel, tuppel_indicies = np.unique(tuppel, return_index=True, axis=0)
        self.knowledge_triplets_list = self.knowledge_triplets_list[tuppel_indicies]
        self.knowledge_nodes_embedding = self.knowledge_nodes_embedding[tuppel_indicies]

        neighborhood_indices = None

        # reduce k
        if len(self.knowledge_triplets_list) < k:
            # extract best trippels
            neighborhood = self.knowledge_triplets_list
            if self.knowledge_indices is not None:
                self.knowledge_indices = self.knowledge_indices[tuppel_indicies]
                neighborhood_indices = self.knowledge_indices

        else:
            # search for best trippels
            # cpu search
            d = self.context_node_embedding.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(np.reshape(self.knowledge_nodes_embedding, (self.knowledge_nodes_embedding.shape[0], -1)))
            D, I = index.search(self.context_node_embedding, k)

            # extract best trippels
            neighborhood = self.knowledge_triplets_list[I[0]]

            # delete extracted trippels
            mask = np.ones(len(self.knowledge_triplets_list), dtype=bool)
            mask[I[0]] = False
            self.knowledge_triplets_list = self.knowledge_triplets_list[mask]
            self.knowledge_nodes_embedding = self.knowledge_nodes_embedding[mask]

            if self.knowledge_indices is not None:
                self.knowledge_indices = self.knowledge_indices[tuppel_indicies]
                neighborhood_indices = self.knowledge_indices[I[0]]
                self.knowledge_indices = self.knowledge_indices[mask]

        return neighborhood, neighborhood_indices

    def new_hop(self, neighborhood, extracted_edges, extracted_edge_indices=None, k=100) -> (list, any):
        """
        :param neighborhood: the last extracted trie neighborhood
        :param extracted_edges: best edges, will creat the subgraph later on
        :param extracted_edge_indices: the corresponding graph indices to extracted_edges (optional)
        :param k: number of edges to be extracted each hop
        :return: the new extracted neighborhood and the updated chosen edges (and indices)
        """

        if neighborhood is not None:
            new_knowledge, new_knowledge_embeddings, new_knowledge_indices = self.pull_from_kg_trie(entities=neighborhood[:, 1])
            self.entities = np.concatenate([self.entities, neighborhood[:, 1]])
        else:
            new_knowledge, new_knowledge_embeddings, new_knowledge_indices = self.pull_from_kg_trie()

        self.update_knowledge(new_knowledge, new_knowledge_embeddings, new_knowledge_indices)

        neighborhood, neighborhood_indices = self.search_neighborhood(k)

        if extracted_edges is None:
            extracted_edges = neighborhood
        else:
            extracted_edges = np.concatenate((extracted_edges, neighborhood), axis=0)

        if neighborhood_indices is not None:
            if extracted_edge_indices is None:
                extracted_edge_indices = neighborhood_indices
            else:
                extracted_edge_indices = np.concatenate((extracted_edge_indices, neighborhood_indices), axis=0)

        return neighborhood, extracted_edges, extracted_edge_indices

    def pull_from_kg_trie(self, entities=None) -> (list, list, any):
        """
        :param entities: the word entities for querying the trie
        :return: the new extracted knowledge from the trie
        """
        if entities is None:
            entities = self.entities

        if type(entities) == type(""):
            entities = [entities]

        new_knowledge_indices = []
        new_knowledge = []
        new_knowledge_embeddings = []

        for entity in entities:
            trippel, embedding, trippel_indices = self.trie.query(entity, avoid_cycles=True)
            new_knowledge += trippel
            new_knowledge_embeddings += embedding
            new_knowledge_indices += trippel_indices

        return new_knowledge, new_knowledge_embeddings, new_knowledge_indices

    def extract_subgraph_from_query(self, n_hops=4, k=100) -> (list, list):
        """
        :param n_hops: number of pulls from the trie
        :param k: how many node are extracted each hop
        :return: the best extracted kg edges and their corresponding indices, if those are given in the trie
        """

        neighborhood, extracted_edges, extracted_edge_indices = None, None, None

        for hop in range(n_hops):
            neighborhood, extracted_edges, extracted_edge_indices = self.new_hop(neighborhood, extracted_edges,
                                                                                 extracted_edge_indices, k=k)

            if extracted_edges is None:
                break

        return extracted_edges, extracted_edge_indices

    def update_knowledge(self, new_knowledge: list, new_embeddings, new_indices=None):
        """This method accumulates the extracted knowledge by adding the newly pulled tire knowledge to
            self.knowledge_triplets_list, self.knowledge_nodes_embedding and self.knowledge_indices"""

        if len(new_knowledge) > 0:
            if self.knowledge_triplets_list is None:
                self.knowledge_triplets_list = np.char.array(new_knowledge)
            else:
                self.knowledge_triplets_list = np.concatenate((self.knowledge_triplets_list, new_knowledge), axis=0)

            if self.knowledge_nodes_embedding is None:
                self.knowledge_nodes_embedding = self.check_emb(new_embeddings)
            else:
                self.knowledge_nodes_embedding = np.concatenate(
                    (self.knowledge_nodes_embedding, self.check_emb(new_embeddings)), axis=0)

            if new_indices is not None:
                if self.knowledge_indices is None:
                    self.knowledge_indices = np.asarray(new_indices)
                else:
                    self.knowledge_indices = np.concatenate(
                        (self.knowledge_indices, np.asarray(new_indices)), axis=0)
