import os
import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from config import ROOT_DIR
from src.preprocess_graph.dataset_builder import build_dataset
from src.medical_hgt.hgt_gat_model import HeteroGraphTransformer

parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')
parser.add_argument('--train_dataset_path', type=str, default='datasets/graph_dataset/train', help='Path of the raw train dataset')
parser.add_argument('--val_dataset_path', type=str, default='datasets/graph_dataset/validation', help='Path of the raw validation dataset')

args = parser.parse_args()


def train(data_loader, edge_weight_regularization):
    model.train()

    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()

        # Forward pass
        answer_preds, edge_weight_dict = model(data)

        # Compute the loss
        loss = criterion(answer_preds, data['answer'].y)

        # Add regularization term for the edge weights
        reg_loss = 0
        for weights in edge_weight_dict.values():
            reg_loss += weights.pow(2).sum()
        loss += edge_weight_regularization * reg_loss

        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(data_loader, edge_weight_regularization):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            answer_preds, edge_weight_dict = model(data)
            loss = criterion(answer_preds, data['answer'].y)

            # Add regularization term for the edge weights
            reg_loss = 0
            for weights in edge_weight_dict.values():
                reg_loss += weights.pow(2).sum()
            loss += edge_weight_regularization * reg_loss

            total_loss += loss.item()

    return total_loss / len(data_loader)


if __name__ == '__main__':

    train_dataset = build_dataset(root_dir=os.path.join(ROOT_DIR, args.train_dataset_path))
    val_dataset = build_dataset(root_dir=os.path.join(ROOT_DIR, args.val_dataset_path))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the model
    model = HeteroGraphTransformer()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Regularization strength for the edge weights
    edge_weight_regularization_strength = 0.001

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(train_loader, edge_weight_regularization_strength)
        val_loss = validate(val_loader, edge_weight_regularization_strength)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
