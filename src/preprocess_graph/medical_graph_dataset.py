import torch
from torch_geometric.data import InMemoryDataset, Data


class MedicalKnowledgeGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MedicalKnowledgeGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Return the names of files in the raw_dir which need to be found in order to skip the download
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        # Return the names of files in the processed_dir which need to be found in order to skip the processing
        return ['data.pt']

    def download(self):
        # Download to self.raw_dir
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # Example of creating a graph with 100 nodes and some random connections.
        edge_index = torch.randint(0, 100, (2, 500))  # 500 random connections
        x = torch.randn(100, 3)  # Node features for each node (e.g., embedding of the medical concepts)

        # For simplicity, randomly generate edge labels (like 'treats', 'causes', etc. in an encoded form)
        edge_attr = torch.randint(0, 10, (500,))  # Just an example

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

        # If you've performed any pre-processing, this is where it would happen.

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
