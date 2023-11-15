import torch
import pickle
import os
import argparse

from src.medical_hgt.model import Model

from config import ROOT_DIR


parser = argparse.ArgumentParser(description='Inference HGT on PrimeKG + Medmcqa')
parser.add_argument('--model_path', type=str, default='experiments/linkneighbor-3.0-4,3,2,10,10,3-128_run1.pth', help='Path of target model to load')
parser.add_argument('--hetero_data_path', type=str, default='datasets/merged_hetero_dataset/processed_graph_480_test_masked_with_edge_uids.pickle', help='Path of the test dataset')

args = parser.parse_args()


def load_model(hetero_data):
    model = Model(hetero_data, hidden_channels=64)

    with open(os.path.join(ROOT_DIR, args.model_path), 'rb') as f:
        saved_object = torch.load(f)
        model.load_state_dict(saved_object.state_dict)

    model.eval()

    return model


def add_and_predict_link(hetero_data, model):

    # Predict links
    with torch.no_grad():
        pred = model(hetero_data)

        eval_pred = pred.detach()
        eval_y = hetero_data["question", "question_answer", "answer"].edge_label.detach()


with open(os.path.join(ROOT_DIR, args.hetero_data_path), 'rb') as f:
    hetero_data = pickle.load(f)

load_model(hetero_data)
