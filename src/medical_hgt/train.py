import time
import copy
import gc
import torch
import numpy as np

from tqdm import tqdm
from torch import cuda
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass

from ml_utils import find_most_relevant_nodes
from src.medical_hgt.llm import LLMFeedback


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


def evaluate_model(llm, medical_hgt, split_loaders, split_name, device, qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping, frac=1.0):
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
    average_llm_context_confidence_list = []
    average_llm_context_accuracy_list = []
    average_llm_vanilla_confidence_list = []
    average_llm_vanilla_accuracy_list = []

    loader = split_loaders[split_name]

    num_batches = round(frac * len(loader))

    print('Validation Batches...')
    for i, batch in enumerate(tqdm(loader)):
        batch_num = i + 1

        batch = batch.to(device)

        with torch.no_grad():

            # Forward pass on the link prediction model
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

            correct_answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            answer_letter_to_op_map = {'A': 'opa', 'B': 'opb', 'C': 'opc', 'D': 'opd'}

            vanilla_accuracy_list, vanilla_confidence_list, context_accuracy_list, context_confidence_list = [], [], [], []
            unseen_questions_indices = batch["question", "question_correct_answer", "answer"].edge_label_index[0]
            if unseen_questions_indices.dim() == 0:
                unseen_questions_indices = unseen_questions_indices.unsqueeze(-1)

            for question_index in unseen_questions_indices:

                question_node_representation = torch.index_select(z_dict['question'], 0, question_index) #z_dict['question'][question_index]

                question_uid = batch['question'].node_uid[question_index].item()
                if question_uid not in llm_feedbacks_dict:
                    continue

                llm_feedback_without_context = llm_feedbacks_dict[question_uid]
                subgraph_tuples = question_to_subgraphs_mapping[question_uid]
                most_relevant_nodes = find_most_relevant_nodes(batch, z_dict, question_node_representation, subgraph_tuples, prime_kg)  # todo
                dataset_row = qa_dataset.iloc[question_uid]
                question_dict = dict(dataset_row.drop(['id', 'cop', 'exp']))
                correct_answer = dataset_row['cop']
                prompt = """Context: {}. Question: {} A. {} B. {} C. {} D. {}""".format(
                    ",".join(most_relevant_nodes),
                    question_dict['question'],
                    question_dict['opa'],
                    question_dict['opb'],
                    question_dict['opc'],
                    question_dict['opd']
                )

                # Process question with context
                output_encodings, predictions = llm.inference(prompt)
                llm_response_dict = llm.get_confidence(correct_answer_map[correct_answer], output_encodings, predictions)
                if llm_response_dict['confidence'] == -1:
                    print(f'Wrong response format. Question {i} ignored during eval')
                    continue

                vanilla_confidence_list.append(llm_feedback_without_context.cop_confidence_without_context)
                vanilla_accuracy_list.append(llm_feedback_without_context.is_correct_without_context)
                context_confidence_list.append(llm_response_dict['cop_confidence'])
                context_accuracy_list.append(llm_response_dict['accuracy'])

                if not llm_feedback_without_context.is_correct_without_context and llm_response_dict['accuracy']:
                    print("\nThe context has helped the LLM!\n")
                    print(f"Question {question_uid}: {question_dict['question']}\n")
                    print(f"LLM's reponse without context: {llm_feedback_without_context.response_without_context}: {question_dict[answer_letter_to_op_map[llm_feedback_without_context.response_without_context]]} --> WRONG!\n")
                    print(f"LLM's reponse with context: {llm_response_dict['response']}: {question_dict[answer_letter_to_op_map[llm_response_dict['response']]]} --> CORRECT!")

            # Calculate average performance of the batch
            batch_average_vanilla_confidence = sum(vanilla_confidence_list) / max(1, len(vanilla_confidence_list))
            batch_average_vanilla_accuracy = sum(vanilla_accuracy_list) / max(1, len(vanilla_accuracy_list))
            batch_average_context_confidence = sum(context_confidence_list) / max(1, len(context_confidence_list))
            batch_average_context_accuracy = sum(context_accuracy_list) / max(1, len(context_accuracy_list))

            if batch_average_context_confidence > 0:
                average_llm_context_confidence_list.append(batch_average_context_confidence)
                average_llm_context_accuracy_list.append(batch_average_context_accuracy)
                average_llm_vanilla_confidence_list.append(batch_average_vanilla_confidence)
                average_llm_vanilla_accuracy_list.append(batch_average_vanilla_accuracy)

        if batch_num >= num_batches:
            break

    medical_hgt.train()

    pos_pred = torch.cat(pos_y_pred_tensors, dim=0).cpu().numpy()
    neg_pred = torch.cat(neg_y_pred_tensors, dim=0).cpu().numpy()
    pos_true = torch.cat(pos_y_true_tensors, dim=0).cpu().numpy()
    neg_true = torch.cat(neg_y_true_tensors, dim=0).cpu().numpy()

    pred = np.concatenate([pos_pred, neg_pred])
    true = np.concatenate([pos_true, neg_true])

    link_prediction_accuracy = roc_auc_score(true, pred)

    llm_results = {
        'vanilla_accuracy': sum(average_llm_vanilla_accuracy_list) / max(1, len(average_llm_vanilla_accuracy_list)),
        'vanilla_confidence': sum(average_llm_vanilla_confidence_list) / max(1, len(average_llm_vanilla_confidence_list)),
        'context_accuracy': sum(average_llm_context_accuracy_list) / max(1, len(average_llm_context_accuracy_list)),
        'context_confidence': sum(average_llm_context_confidence_list) / max(1, len(average_llm_context_confidence_list))
    }

    return link_prediction_accuracy, llm_results


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

    llm_results: dict


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

    llm_results: dict

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
    pos_labels = pos_labels.view(-1).float()
    pos_loss = F.binary_cross_entropy_with_logits(pos_preds, pos_labels)

    # Calculate loss for negative predictions
    neg_labels = neg_labels.view(-1).float()
    neg_loss = F.binary_cross_entropy_with_logits(neg_preds, neg_labels)

    # Combine the losses
    total_loss = pos_loss + neg_loss / 3.0

    return total_loss


def train_model(llm, medical_hgt, split_loaders, device, file_name, qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping, num_epochs=30, lr=0.001, link_prediction_loss_weight=0.3):
    medical_hgt = medical_hgt.to(device)

    medical_hgt.train()

    opt = torch.optim.Adam(medical_hgt.parameters(), lr=lr)

    start_time = get_time()
    print(f'start time: {start_time}; will save results to {file_name}')

    train_loader = split_loaders['train']

    llm_relevancy_loss_weight = 1 - link_prediction_loss_weight

    epoch_results = []

    for epoch_num in range(1, num_epochs + 1):
        train_start_time = get_time()

        train_losses = []
        pos_y_pred_tensors = []
        neg_y_pred_tensors = []
        pos_y_true_tensors = []
        neg_y_true_tensors = []

        num_batches = len(train_loader)

        print("Train Batches...")
        for batch in tqdm(train_loader):
            batch = batch.to(device)

            opt.zero_grad()

            pos_train_pred, neg_train_pred, z_dict = medical_hgt(batch)

            pos_train_y = batch["question", "question_correct_answer", "answer"].edge_label.squeeze()
            neg_train_y = batch["question", "question_wrong_answer", "answer"].edge_label.squeeze()

            if pos_train_y.dim() == 0:
                pos_train_y = pos_train_y.view(1)

            if neg_train_y.dim() == 0:
                neg_train_y = neg_train_y.view(1)

            link_prediction_loss = compute_link_prediction_loss(pos_train_pred, neg_train_pred, pos_train_y, neg_train_y)

            llm_relevancy_loss = compute_llm_relevancy_loss(batch, z_dict, llm_feedbacks_dict)

            total_loss = link_prediction_loss_weight * link_prediction_loss + llm_relevancy_loss_weight * llm_relevancy_loss
            total_loss.backward()

            opt.step()

            pos_y_pred_tensors.append(pos_train_pred.detach())
            neg_y_pred_tensors.append(neg_train_pred.detach())
            pos_y_true_tensors.append(pos_train_y.detach().long())
            neg_y_true_tensors.append(neg_train_y.detach().long())

            train_losses.append(total_loss.detach().item())

        train_end_time = get_time()

        pos_pred = torch.cat(pos_y_pred_tensors, dim=0).cpu().numpy()
        neg_pred = torch.cat(neg_y_pred_tensors, dim=0).cpu().numpy()
        pos_true = torch.cat(pos_y_true_tensors, dim=0).cpu().numpy()
        neg_true = torch.cat(neg_y_true_tensors, dim=0).cpu().numpy()

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
        link_prediction_val_acc, val_llm_results = evaluate_model(llm, medical_hgt, split_loaders, 'val', device, qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping)

        combined_val_acc = link_prediction_loss_weight * link_prediction_val_acc + llm_relevancy_loss_weight * val_llm_results['context_accuracy']

        epoch_result = EpochResult(
            epoch_num=epoch_num,
            train_start_time=train_start_time,
            train_end_time=train_end_time,
            mean_train_loss=round(np.mean(train_losses), 4),
            train_acc=round(train_acc, 4),
            val_acc=round(combined_val_acc, 4),
            llm_results=val_llm_results
        )

        epoch_results.append(epoch_result)
        print(f'\r{epoch_result}')

    state_dict = copy.deepcopy(medical_hgt.state_dict())
    test_acc, test_llm_results = evaluate_model(llm, medical_hgt, split_loaders, 'test', device, qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping)

    medical_hgt.eval()

    end_time = get_time()

    combined_test_acc = link_prediction_loss_weight * test_acc + llm_relevancy_loss_weight * test_llm_results['context_accuracy']
    medical_hgt_result = ModelResult(start_time, end_time, epoch_results, state_dict, round(test_acc, 4), test_llm_results)
    torch.save(medical_hgt_result, file_name)

    train_time_min = medical_hgt_result.get_total_train_time_min()
    print(f'\rTest Accuracy: {combined_test_acc:.3f}; LLM Results: {test_llm_results}, Total Train Time: {train_time_min} min')

    return medical_hgt_result


def compute_llm_relevancy_loss(batch, z_dict, llm_feedback_dict: LLMFeedback):
    # Initialize loss
    loss = 0.0
    num_nodes = 0

    # Iterate over all nodes to form triplets and compute loss
    for question_index, question_uid in enumerate(batch['question'].node_uid):

        if question_uid.item() not in llm_feedback_dict:
            continue

        llm_feedback = llm_feedback_dict[question_uid.item()]
        question_representation = z_dict['question'][question_index]
        confidence_without_context = llm_feedback.cop_confidence_without_context
        for node_type, confidence_dict in llm_feedback.cop_confidences_with_context.items():

            if len(batch[node_type]) == 0:
                continue

            for node_id, confidence_with_context in confidence_dict.items():
                batch_node_index = torch.where(batch[node_type].node_uid == node_id)[0][0]
                current_node_representation = torch.index_select(z_dict[node_type], 0, batch_node_index)

                # Calculate the distance between the node embedding and the question node embedding
                distance = torch.norm(current_node_representation - question_representation, p=2)

                conf_diff = confidence_with_context - confidence_without_context

                # For positive relevance, penalize being far from the question node
                # For negative relevance, penalize being close to the question node
                if conf_diff > 0:
                    weighted_loss = conf_diff * distance
                elif conf_diff < 0:
                    # Invert the distance measure for negative relevance
                    weighted_loss = -conf_diff * (1 / (distance + 1e-6))  # adding a small constant to avoid division by zero
                else:  # relevance is around 0, neutral
                    weighted_loss = 0

                # Accumulate the loss
                loss += weighted_loss

                num_nodes += 1

    return loss / max(num_nodes, 1)
