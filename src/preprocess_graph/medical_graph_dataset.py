import re
import torch
import os
import pickle

from torch_geometric.data import InMemoryDataset


class MedicalKnowledgeGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        path = os.path.join(self.root, "raw")
        return os.listdir(path)

    @property
    def processed_file_names(self):
        # Define the name of the processed file (where processed data will be saved).
        return self.raw_file_names

    def download(self):
        # You can skip this method if your data is already available and you don't need to download it.
        pass

    def process(self):
        pattern = r'part_(\d+)\.pickle'
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with open(raw_path, 'rb') as f:
                hetero_data = pickle.load(f)

            file_name = re.search(pattern, raw_path)

            torch.save(hetero_data, os.path.join(self.processed_dir, f'{file_name}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        with open(os.path.join(self.processed_dir, f'part_{idx}.pickle', 'rb')) as f:
            hetero_data = pickle.load(f)

        return hetero_data
