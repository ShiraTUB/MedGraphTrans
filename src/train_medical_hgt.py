import argparse

from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch_geometric.loader import DataLoader

from src.medical_hgt.model import MedicalQAModel
from src.preprocess_graph.medical_graph_dataset import MedicalKnowledgeGraphDataset


parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')

parser.add_argument('--data_dir', type=str, default='/datasets/OGB_MAG.pk', help='Preprocessed graph path')
parser.add_argument('--model_dir', type=str, default='./hgt_models', help='Trained model save path')
parser.add_argument('--plot', action='store_true', help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs to run')
parser.add_argument('--train_dataset_path', type=str, default='/datasets/processed_graph_dataset/train.pk', help='Path of the processed train dataset')
parser.add_argument('--val_dataset_path', type=str, default='/datasets/processed_graph_dataset/val.pk', help='Path of the processed val dataset')
parser.add_argument('--test_dataset_path', type=str, default='/datasets/processed_graph_dataset/test.pk', help='Path of the processed test dataset')

args = parser.parse_args()


def train():
    train_data = MedicalKnowledgeGraphDataset(root=args.train_dataset_path)
    val_data = MedicalKnowledgeGraphDataset(root=args.val_dataset_path)

    med_qa_model = MedicalQAModel()

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
