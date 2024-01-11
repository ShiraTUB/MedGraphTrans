import os
import argparse
from dataclasses import dataclass

from src.medical_hgt.model import MedicalHGT
from src.medical_hgt.train import train
from config import ROOT_DIR

parser = argparse.ArgumentParser(description='Training HGT on PrimeKG + Medmcqa')
parser.add_argument('--experiment_output_path', type=str, default='experiments', help='Path of the target experiments folder')
parser.add_argument('--prime_kg_dataset', type=str, default='datasets/primeKG_nx_medium.pickle', help='PrimeKG pickle path')
parser.add_argument('--train_dataset', type=str, default='datasets/train_mini_batches_32_cpu.pickle', help='Processed train data pickle path')
parser.add_argument('--val_dataset', type=str, default='datasets/val_mini_batches_32_cpu.pickle', help='Processed val data pickle path')
parser.add_argument('--test_dataset', type=str, default='datasets/test_mini_batches_cpu_32.pickle', help='Processed test data pickle path')
parser.add_argument('--train_llm_feedback', type=str, default='datasets//llm_feedbacks_train.pickle', help='Processed llm feedback (train set)')
parser.add_argument('--val_llm_feedback', type=str, default='datasets/llm_feedback/llm_feedbacks_validation.pickle', help='Processed validation llm feedback (validation set)')
parser.add_argument('--kg_train_subgraphs_mapping', type=str, default='datasets/subgraphs_dict_train.pickle', help='Mapping between train question ids and their respective subgraphs')
parser.add_argument('--kg_val_subgraphs_mapping', type=str, default='datasets/subgraphs_dict_val.pickle', help='Mapping between validation question ids and their respective subgraphs')

args = parser.parse_args()


@dataclass(frozen=True)
class ExperimentsParams:
    """Helper class for holding the parameters of a training experiment."""
    num_epochs: int
    lr: float
    channels: int
    num_heads: int
    num_layers: int

    def get_file_name(self):
        """Generates the file name for storing the results, based on the params."""
        folder_path = os.path.join(ROOT_DIR, 'experiments')
        return f'{folder_path}/experiment-{self.num_epochs}_epochs-{self.lr}_lr-{self.channels}_channels-{self.num_heads}_head-{self.num_layers}_layers'


def run_experiments(experiment_params, device, llm, train_llm_feedbacks_dict, val_llm_feedbacks_dict, question_to_subgraphs_mapping, prime_kg, qa_dataset, data_loaders):
    """Runs a multi-trial experiment using the given ExperimentsParams."""

    file_name = f'{experiment_params.get_file_name()}.pth'
    medical_hgt = MedicalHGT(channels=experiment_params.channels, num_heads=experiment_params.num_heads, num_layers=experiment_params.num_layers)
    medical_hgt_result = train(llm=llm,
                               medical_hgt=medical_hgt,
                               split_loaders=data_loaders,
                               device=device,
                               file_name=file_name,
                               qa_dataset=qa_dataset,
                               prime_kg=prime_kg,
                               train_llm_feedbacks_dict=train_llm_feedbacks_dict,
                               val_llm_feedbacks_dict=val_llm_feedbacks_dict,
                               question_to_subgraphs_mapping=question_to_subgraphs_mapping,
                               num_epochs=experiment_params.num_epochs,
                               lr=experiment_params.lr)

    return medical_hgt_result


if __name__ == '__main__':
    experiment = ExperimentsParams(num_epochs=50, lr=0.001, channels=32, num_heads=2, num_layers=1)

    # Load data and run experiment

