import openai
import os
import re
import pickle

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from src.preprocess_graph.build_subgraph import process_raw_graph_data
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class MedicalKnowledgeGraphDataset(InMemoryDataset):
    def __init__(self, root, purpose, transform=None, pre_transform=None):
        self.filename = purpose
        #            embeddings = openai.Embedding.create(input=[self.text], model=model)['data'][0]['embedding']
        super(MedicalKnowledgeGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        path = os.path.join(self.root, "raw")
        filenames = os.listdir(path)
        raw_path_list = []

        for filename in filenames:
            raw_path_list.append(filename)

        return raw_path_list

    def get_max_processed_index(self):
        maximum_index = 0

        for index, path in enumerate(self.raw_paths):
            str_index_list = re.findall(r'\d+', path)
            int_index_list = [int(str_int) for str_int in str_index_list]
            maximum_index = max(maximum_index, max(int_index_list))

        return maximum_index

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        max_index = self.get_max_processed_index()
        return [f'part_{i}.pickle' for i in range(max_index + 1)]

    def download(self):
        pass

    def process(self):
        index = 0

        for _, path in tqdm(enumerate(self.raw_paths), total=len(self.raw_paths)):
            raw_graph = pickle.load(open(path, 'rb'))

            processed_graph = process_raw_graph_data(raw_graph)

            pickle.dump(processed_graph, open(os.path.join(self.processed_dir, self.filename, f'part_{index}.pickle'), "wb"))
            index += 1
