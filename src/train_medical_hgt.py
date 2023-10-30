import argparse
import os
import torch
import copy

from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch_geometric.loader import DataLoader

from src.medical_hgt.model import MedicalQAModel
from src.medical_hgt.encoder import MedicalEncoder
from src.medical_hgt.decoder import Decoder, PositionalEncoding, Generator
from src.preprocess_graph.medical_graph_dataset import MedicalKnowledgeGraphDataset
from config import ROOT_DIR


parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')

parser.add_argument('--data_dir', type=str, default='/datasets/OGB_MAG.pk', help='Preprocessed graph path')
parser.add_argument('--model_dir', type=str, default='./hgt_models', help='Trained model save path')
parser.add_argument('--plot', action='store_true', help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs to run')
parser.add_argument('--train_dataset_path', type=str, default='datasets/graph_dataset/train', help='Path of the raw train dataset')
parser.add_argument('--val_dataset_path', type=str, default='datasets/graph_dataset/val', help='Path of the raw val dataset')
parser.add_argument('--test_dataset_path', type=str, default='datasets/graph_dataset/test', help='Path of the raw test dataset')
parser.add_argument('--d_model', type=int, default=768, help='TODO')
parser.add_argument('--d_ff', type=int, default=1024, help='TODO')
parser.add_argument('--n_heads', type=int, default=4, help='TODO')
parser.add_argument('--dropout', type=float, default=0.1, help='TODO')
parser.add_argument('--n_blocks', type=int, default=4, help='TODO')
parser.add_argument('--lr', type=float, default=3e-4, help='TODO')

args = parser.parse_args()


def train():
    train_data = MedicalKnowledgeGraphDataset(root=os.path.join(ROOT_DIR, args.train_dataset_path), purpose='raw')
    val_data = MedicalKnowledgeGraphDataset(root=os.path.join(ROOT_DIR, args.train_dataset_path), purpose='val')

    vocab = weights.shape[0]

    encoder = MedicalEncoder(args.d_model, n_types, metadata)

    embs_word = torch.nn.Embedding.from_pretrained(weights, padding_idx=0, freeze=False)

    pos = PositionalEncoding(d_model=weights.shape[1], dropout=args.dropout)

    decoder = Decoder(d_model=768, d_ff=args.d_ff, n_heads=args.n_heads, dropout=0.1, n_blocks=args.n_blocks)

    tgt_emb = torch.nn.Sequential(embs_word, copy.deepcopy(pos))

    generator = Generator(args.d_model, vocab)

    med_qa_model = MedicalQAModel(encoder, decoder, tgt_emb, generator, args.lr)

    train_loader = DataLoader(train_data, num_workers=8)
    val_loader = DataLoader(val_data, num_workers=8)

    trainer = Trainer(accelerator="cpu", devices="auto", max_epochs=2)
    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(med_qa_model, train_loader, val_loader, max_lr=1e-2, num_training=100)
    med_qa_model.learning_rate = lr_finder.suggestion()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = TensorBoardLogger(save_dir='train_dir', name='training_name')

    val_check_interval = len(train_data)
    limit_val_batches = len(val_data)

    trainer = Trainer(
        limit_val_batches=limit_val_batches,
        log_every_n_steps=50,
        val_check_interval=val_check_interval,
        max_epochs=100,
        default_root_dir='train_dir',
        callbacks=[lr_monitor],
        logger=logger,
        accelerator="auto",
        devices="auto"
    )

    trainer.fit(med_qa_model, train_loader, val_loader)


if __name__ == "__main__":
    train()
