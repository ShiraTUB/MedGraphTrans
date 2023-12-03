import time
import copy
import gc

import torch
import numpy as np

from torch import cuda
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass

from ml_utils import find_subgraph_bfs, find_most_relevant_nodes


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


def evaluate_model(medical_hgt, split_loaders, split_name, device, prime_kg, frac=1.0):
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
    medical_hgt.eval()

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
            pos_pred, neg_pred, z_dict = medical_hgt(batch)

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

            knowledge_nodes_per_question_dict = {}
            for node_index, question_node_representation in enumerate(z_dict['question']):
                subgraph_nodes_uid_dict = find_subgraph_bfs(batch, node_index, 'question')
                question_node_uid = batch['question'].node_uid[node_index]
                most_relevant_nodes = find_most_relevant_nodes(batch, z_dict, question_node_representation, subgraph_nodes_uid_dict, prime_kg)
                knowledge_nodes_per_question_dict[question_node_uid] = most_relevant_nodes

        if batch_num >= num_batches:
            break

    medical_hgt.train()

    pos_pred = torch.cat(pos_y_pred_tensors, dim=0).numpy()
    neg_pred = torch.cat(neg_y_pred_tensors, dim=0).numpy()
    pos_true = torch.cat(pos_y_true_tensors, dim=0).numpy()
    neg_true = torch.cat(neg_y_true_tensors, dim=0).numpy()

    pred = np.concatenate([pos_pred, neg_pred])
    true = np.concatenate([pos_true, neg_true])

    return roc_auc_score(true, pred), knowledge_nodes_per_question_dict


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


def compute_link_prediction_loss(pos_preds: torch.Tensor, neg_preds: torch.Tensor, pos_labels: torch.Tensor, neg_labels: torch.Tensor) -> torch.Tensor:
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


def train_model(medical_hgt, llm, split_loaders, device, file_name, qa_dataset, prime_kg, num_epochs=30, lr=0.001):
    medical_hgt = medical_hgt.to(device)
    llm = llm.to(device)

    medical_hgt.train()
    llm.train()

    opt = torch.optim.Adam(medical_hgt.parameters(), lr=lr)

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

            # internally, the medical_hgt is applied using all the batch's edges (i.e.,
            # batch.edge_index) but only outputs predictions on edges to be labeled
            # (i.e., batch.edge_label_index).
            pos_train_pred, neg_train_pred, z_dict = medical_hgt(batch)

            pos_train_y = batch["question", "question_correct_answer", "answer"].edge_label.squeeze()
            neg_train_y = batch["question", "question_wrong_answer", "answer"].edge_label.squeeze()

            if pos_train_y.dim() == 0:
                pos_train_y = pos_train_y.view(1)

            if neg_train_y.dim() == 0:
                neg_train_y = neg_train_y.view(1)

            confidence_diffs_per_question = []  # a list of tuples (question_embeddings, conf_diffs_dict_per_question)
            # compute the llm's feedback per question in the batch
            for node_index, question_node_representation in enumerate(z_dict['question']):
                qa_index = batch['question'].node_uid[node_index].item()
                subgraph_nodes_uid_dict = find_subgraph_bfs(batch, node_index, 'question')
                prompt_dict = dict(qa_dataset.iloc[qa_index].drop(['id', 'cop', 'exp']))
                correct_answer = qa_dataset.iloc[qa_index]['cop']
                current_confidence_diffs_dict = llm(subgraph_nodes_uid_dict, prime_kg, prompt_dict, correct_answer)
                confidence_diffs_per_question.append((question_node_representation, current_confidence_diffs_dict))
                break

            link_prediction_loss = compute_link_prediction_loss(pos_train_pred, neg_train_pred, pos_train_y, neg_train_y)
            llm_relevancy_loss = compute_llm_relevancy_loss(batch, z_dict, confidence_diffs_per_question)

            total_loss = (link_prediction_loss + (llm_relevancy_loss * 0.01))
            total_loss.backward()

            opt.step()

            pos_y_pred_tensors.append(pos_train_pred.detach())
            neg_y_pred_tensors.append(neg_train_pred.detach())
            pos_y_true_tensors.append(pos_train_y.detach().long())
            neg_y_true_tensors.append(neg_train_y.detach().long())

            train_losses.append(total_loss.detach().item())
            break

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
        # give the medical_hgt a "second try" for earlier batches, for which it couldn't
        # have yet applied anything it learned in later batches.
        train_acc = roc_auc_score(true, pred)

        # The validation ROC AUC is computed by running through the validation set
        # at the end of every epoch.
        val_acc, val_most_relevant_nodes = evaluate_model(medical_hgt, split_loaders, 'val', device, prime_kg=prime_kg)

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

    state_dict = copy.deepcopy(medical_hgt.state_dict())
    test_acc, test_most_relevant_nodes = evaluate_model(medical_hgt, split_loaders, 'test', device, prime_kg=prime_kg)

    medical_hgt.eval()

    end_time = get_time()
    medical_hgt_result = ModelResult(start_time, end_time, epoch_results, state_dict, round(test_acc, 4))
    torch.save(medical_hgt_result, file_name)

    train_time_min = medical_hgt_result.get_total_train_time_min()
    print(f'\rTest Accuracy: {test_acc:.3f}; Total Train Time: {train_time_min} min')

    return medical_hgt_result


def compute_llm_relevancy_loss(batch, z_dict, gradients_per_questions_list):
    # Initialize loss
    loss = 0.0
    num_nodes = 0

    # Iterate over all nodes to form triplets and compute loss
    for question_embedding, grads_dict in gradients_per_questions_list:
        for node_type, grad_info_dict in grads_dict.items():
            batch_node_indices = [torch.where(batch[node_type].node_uid == x)[0][0] for x in list(grad_info_dict.keys())]
            gradients_list = list(grad_info_dict.values())
            for i, node_index in enumerate(batch_node_indices):
                current_node_embedding = torch.index_select(z_dict[node_type], 0, node_index)

                # Calculate the distance between the node embedding and the central node embedding
                distance = torch.norm(current_node_embedding - question_embedding, p=2)

                # Determine the weight based on relevance
                relevance = gradients_list[i]

                # For positive relevance, penalize being far from the central node
                # For negative relevance, penalize being close to the central node
                if relevance > 0:
                    weighted_loss = relevance * distance
                elif relevance < 0:
                    # Invert the distance measure for negative relevance
                    weighted_loss = -relevance * (1 / (distance + 1e-6))  # adding a small constant to avoid division by zero
                else:  # relevance is around 0, neutral
                    weighted_loss = 0

                # Accumulate the loss
                loss += weighted_loss

            num_nodes += len(gradients_list)

    return loss / num_nodes
