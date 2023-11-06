import os.path
import pickle


def build_dataset(root_dir: str):
    data_list = []
    file_names_list = os.listdir(root_dir)
    for file_name in file_names_list:
        path = os.path.join(root_dir, file_name)
        with open(path, 'rb') as f:
            hetero_data = pickle.load(f)
            data_list.append(hetero_data)
    return data_list
