import openai
import torch
import heapq
import time

import torch.nn.functional as F

from dataclasses import dataclass

from config import OPENAI_API_KEY
from src.medical_hgt.llm import LLMFeedback

openai.api_key = OPENAI_API_KEY


def find_most_relevant_nodes(batch, z_dict, question_nodes_embedding, subgraph_tuples, prime_gk, k=1):
    """

    Args:
        batch: a batch of HeteroData from the training dataset of MedGraphTrans
        z_dict: the node representations after an HGT forward pass. ({node_type: tensor, node_type: tenser...})
        question_nodes_embedding: The representation of the question node
        subgraph_tuples: a list of tuples (node_uid, node_type) from the question's heterogeneous graph
        prime_gk: the knowledge graph used for knowledge extraction
        k: number of relevant nodes to return

    Returns:
        relevant_nodes_list: the list of all relevant nodes

    """
    # Using a heap for efficient minimum distance tracking
    relevant_nodes_heap = []
    heapq.heapify(relevant_nodes_heap)

    for node_uid, node_type in subgraph_tuples:

        if len(batch[node_type]) == 0:
            continue
        node_index = torch.where(batch[node_type].node_uid == node_uid)[0][0]
        node_embeddings = z_dict[node_type][node_index]

        # Calculate distance
        distance = torch.norm(question_nodes_embedding - node_embeddings, p=2)

        # Get node information
        node_info = prime_gk.nodes[node_uid]
        node_info_string = f"The {node_info['type']} {node_info['name']}"

        # Update the relevant nodes list and distances
        if len(relevant_nodes_heap) < k:
            heapq.heappush(relevant_nodes_heap, (distance, node_info_string))
        else:
            # Only update if the current distance is greater than the smallest in the heap
            if distance > relevant_nodes_heap[0][0]:
                heapq.heappop(relevant_nodes_heap)
                heapq.heappush(relevant_nodes_heap, (distance, node_info_string))

    # Convert heap to a list of relevant nodes, sorting by distance
    relevant_nodes_list = [node for _, node in sorted(relevant_nodes_heap, reverse=True)]

    return relevant_nodes_list


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

    # accuracy of the link prediction task on the training/validation set at the end of this epoch
    train_roc_aoc: float
    train_precision: float
    train_recall: float
    train_f1: float

    val_roc_aoc: float
    val_precision: float
    val_recall: float
    val_f1: float

    # accuracy of the relevancy scoring task at the end of this epoch
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

    # final accuracy on the test set (after all epochs)
    test_roc_aoc: float
    test_precision: float
    test_recall: float
    test_f1: float

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


def compute_link_prediction_loss(pos_preds: torch.Tensor, neg_preds: torch.Tensor, pos_labels: torch.Tensor, neg_labels: torch.Tensor, device) -> torch.Tensor:
    """
    Args:
    pos_preds (torch.Tensor): Predictions for positive links, expected to be logits.
    neg_preds (torch.Tensor): Predictions for negative links, expected to be logits.
    pos_labels (torch.Tensor): Ground truth labels for positive links.
    neg_labels (torch.Tensor): Ground truth labels for negative links.

    Returns:
    torch.Tensor: The combined binary cross-entropy loss for positive and negative predictions.
    """

    # Assign weights
    pos_weight = torch.tensor([2.0])  # As we have 2-3 times more negative samples
    neg_weight = torch.tensor([1.0])

    # Calculate loss for positive predictions
    pos_labels = pos_labels.view(-1).float()
    pos_loss = F.binary_cross_entropy_with_logits(pos_preds.to(device), pos_labels.to(device), weight=pos_weight.to(device))

    # Calculate loss for negative predictions
    neg_labels = neg_labels.view(-1).float()
    neg_loss = F.binary_cross_entropy_with_logits(neg_preds.to(device), neg_labels.to(device), weight=neg_weight.to(device))

    # Combine the losses
    total_loss = (pos_loss + neg_loss) / 2

    return total_loss


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
