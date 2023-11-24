import time
import copy
import gc

import torch
import numpy as np

from torch import cuda
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass

from ml_utils import decode_edge_weights


def free_memory():
    """Clears the GPU cache and triggers garbage collection, to reduce OOMs."""
    cuda.empty_cache()
    gc.collect()


def get_masked(target, batch, split_name):
    """
    Applies the mask for a given split but no-ops if the mask isn't present.

    This is useful for shared models where the data may or may not be masked.
    """
    mask_name = f'{split_name}_mask'
    return target if mask_name not in batch else target[batch[mask_name]]


def evaluate_model(model, split_loaders, split_name, device, prime_kg, frac=1.0):
    """
    Args:
    model (torch.nn.Module): The model to evaluate.
    split_loaders (dict): A dictionary containing the data loaders for different splits.
    split_name (str): The name of the split to evaluate (e.g., 'val', 'test').
    device (torch.device): The device to run the model on.
    frac (float): Fraction of the dataset to use for evaluation.

    Returns:
    float: The ROC AUC score for the evaluated split.
    """
    model.eval()

    pos_y_true_tensors = []
    neg_y_true_tensors = []
    pos_y_pred_tensors = []
    neg_y_pred_tensors = []

    loader = split_loaders[split_name]
    num_batches = round(frac * len(loader))

    for i, batch in enumerate(loader):
        batch_num = i + 1
        print(f'\r{split_name} batch {batch_num} / {num_batches}', end='')

        batch = batch.to(device)

        with torch.no_grad():
            pos_pred, neg_pred = model(batch)

            most_relevant_edges = decode_edge_weights(model.edge_weights_dict, batch.edge_index_dict, prime_kg)

            pos_eval_y = batch["question", "question_correct_answer", "answer"].edge_label.squeeze()
            neg_eval_y = batch["question", "question_wrong_answer", "answer"].edge_label.squeeze()

            if pos_eval_y.dim() == 0:
                pos_eval_y = pos_eval_y.view(1)

            if neg_eval_y.dim() == 0:
                neg_eval_y = neg_eval_y.view(1)

            pos_y_pred_tensors.append(pos_pred.detach())
            neg_y_pred_tensors.append(neg_pred.detach())
            pos_y_true_tensors.append(pos_eval_y.detach())
            neg_y_true_tensors.append(neg_eval_y.detach())

        if batch_num >= num_batches:
            break

    model.train()

    pos_pred = torch.cat(pos_y_pred_tensors, dim=0).numpy()
    neg_pred = torch.cat(neg_y_pred_tensors, dim=0).numpy()
    pos_true = torch.cat(pos_y_true_tensors, dim=0).numpy()
    neg_true = torch.cat(neg_y_true_tensors, dim=0).numpy()

    pred = np.concatenate([pos_pred, neg_pred])
    true = np.concatenate([pos_true, neg_true])

    return roc_auc_score(true, pred)


@dataclass(frozen=True)
class EpochResult:
    # "index" of the epoch
    # (this is also discernable from the position in ModelResult.epoch_results)
    epoch_num: int

    # Unix timestamps (seconds) when the epoch started/finished training, but not
    # counting evaluation
    train_start_time: int
    train_end_time: int

    # mean train loss taken across all batches
    mean_train_loss: float

    # accuracy on the training/validation set at the end of this epoch
    train_acc: float
    val_acc: float


@dataclass(frozen=True)
class ModelResult:
    # Unix timestamp for when the model started training
    start_time: int
    # Unix timestamp for when the model completely finished (including evaluation
    # on the test set)
    end_time: int

    # list of EpochResults -- see above
    epoch_results: list

    # model state for reloading
    state_dict: dict

    # final accuracy on the full test set (after all epochs)
    test_acc: float

    def get_total_train_time_sec(self):
        """
        Helper function for calculating the total amount of time spent training, not
        counting evaluation. In other words, this only counts the forward pass, the
        loss calculation, and backprop for each batch.
        """
        return sum([
            er.train_end_time - er.train_start_time
            for er in self.epoch_results])

    def get_total_train_time_min(self):
        """get_total_train_time_sec, converted to minutes. See above."""
        return self.get_total_train_time_sec() // 60


def get_time():
    """Returns the current Unix (epoch) timestamp, in seconds."""
    return round(time.time())


def compute_loss(pos_preds: torch.Tensor, neg_preds: torch.Tensor, pos_labels: torch.Tensor, neg_labels: torch.Tensor) -> torch.Tensor:
    """
    Args:
    pos_preds (torch.Tensor): Predictions for positive links, expected to be logits.
    neg_preds (torch.Tensor): Predictions for negative links, expected to be logits.
    pos_labels (torch.Tensor): Ground truth labels for positive links.
    neg_labels (torch.Tensor): Ground truth labels for negative links.

    Returns:
    torch.Tensor: The combined binary cross-entropy loss for positive and negative predictions.
    """

    # Calculate loss for positive predictions
    pos_loss = F.binary_cross_entropy_with_logits(pos_preds, pos_labels.view(-1).float())

    # Calculate loss for negative predictions
    neg_loss = F.binary_cross_entropy_with_logits(neg_preds, neg_labels.view(-1).float())

    # Combine the losses
    total_loss = pos_loss + (neg_loss / 3)

    return total_loss


def train_model(model, split_loaders, device, file_name, prime_kg, num_epochs=30, lr=0.1):
    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = get_time()
    print(f'start time: {start_time}; will save results to {file_name}')

    train_loader = split_loaders['train']
    epoch_results = []

    for epoch_num in range(1, num_epochs + 1):
        train_start_time = get_time()

        train_losses = []
        pos_y_pred_tensors = []
        neg_y_pred_tensors = []
        pos_y_true_tensors = []
        neg_y_true_tensors = []

        num_batches = len(train_loader)

        for i, batch in enumerate(train_loader):
            batch_num = i + 1

            # this is a carriage return trick for overwriting past lines
            print(f'\rEpoch {epoch_num}: batch {batch_num} / {num_batches}', end='')

            opt.zero_grad()
            batch = batch.to(device)

            # internally, the model is applied using all the batch's edges (i.e.,
            # batch.edge_index) but only outputs predictions on edges to be labeled
            # (i.e., batch.edge_label_index).
            pos_train_pred, neg_train_pred = model(batch)

            pos_train_y = batch["question", "question_correct_answer", "answer"].edge_label.squeeze()
            neg_train_y = batch["question", "question_wrong_answer", "answer"].edge_label.squeeze()

            if pos_train_y.dim() == 0:
                pos_train_y = pos_train_y.view(1)

            if neg_train_y.dim() == 0:
                neg_train_y = neg_train_y.view(1)

            loss = compute_loss(pos_train_pred, neg_train_pred, pos_train_y, neg_train_y)
            loss.backward()
            opt.step()

            # Retrieve learned weights from previous batch
            for edge_type in model.edge_weights_dict.keys():
                if edge_type in model.selected_weights_dict:
                    model.edge_weights_dict[edge_type][model.relevant_edge_indices_dict[edge_type]] = model.selected_weights_dict[edge_type]
                    del model.relevant_edge_indices_dict[edge_type]
                    del model.selected_weights_dict[edge_type]

            # for edge_type, edge_indices in model.hgt.relevant_edge_weights_indices_dict.items():
            #     if edge_type in model.relevant_weights_dict:
            #         for i, weight_index in enumerate(edge_indices):
            #             temp = model.edge_weights_dict[edge_type][0].clone()
            #             temp[weight_index] = model.relevant_weights_dict[edge_type][i][0]
            #             model.edge_weights_dict[edge_type][0] = temp

            pos_y_pred_tensors.append(pos_train_pred.detach())
            neg_y_pred_tensors.append(neg_train_pred.detach())
            pos_y_true_tensors.append(pos_train_y.detach().long())
            neg_y_true_tensors.append(neg_train_y.detach().long())

            train_losses.append(loss.detach().item())

        train_end_time = get_time()

        pos_pred = torch.cat(pos_y_pred_tensors, dim=0).numpy()
        neg_pred = torch.cat(neg_y_pred_tensors, dim=0).numpy()
        pos_true = torch.cat(pos_y_true_tensors, dim=0).numpy()
        neg_true = torch.cat(neg_y_true_tensors, dim=0).numpy()

        pred = np.concatenate([pos_pred, neg_pred])
        true = np.concatenate([pos_true, neg_true])

        # the training ROC AUC is computed using all the predictions (and ground
        # truth labels) made during the entire epoch, across all batches. Note that
        # this is arguably a bit inconsistent with validation below since it doesn't
        # give the model a "second try" for earlier batches, for which it couldn't
        # have yet applied anything it learned in later batches.
        train_acc = roc_auc_score(true, pred)

        # The validation ROC AUC is computed by running through the validation set
        # at the end of every epoch.
        val_acc = evaluate_model(model, split_loaders, 'val', device, prime_kg=prime_kg)

        epoch_result = EpochResult(
            epoch_num=epoch_num,
            train_start_time=train_start_time,
            train_end_time=train_end_time,
            mean_train_loss=round(np.mean(train_losses), 4),
            train_acc=round(train_acc, 4),
            val_acc=round(val_acc, 4)
        )

        epoch_results.append(epoch_result)
        print(f'\r{epoch_result}')

    state_dict = copy.deepcopy(model.state_dict())
    test_acc = evaluate_model(model, split_loaders, 'test', device)

    model.eval()

    end_time = get_time()
    model_result = ModelResult(start_time, end_time, epoch_results, state_dict, round(test_acc, 4))
    torch.save(model_result, file_name)

    train_time_min = model_result.get_total_train_time_min()
    print(f'\rTest Accuracy: {test_acc:.3f}; Total Train Time: {train_time_min} min')

    return model_result
