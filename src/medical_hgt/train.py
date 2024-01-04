import copy
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import BinaryRecall, BinaryF1Score, BinaryPrecision

from src.medical_hgt.ml_utils import find_most_relevant_nodes, EpochResult, ModelResult, get_time, compute_llm_relevancy_loss, compute_link_prediction_loss


def train(llm,
          medical_hgt,
          split_loaders,
          device,
          file_name,
          qa_dataset,
          prime_kg,
          llm_feedbacks_dict,
          question_to_subgraphs_mapping,
          num_epochs=30,
          lr=0.001,
          link_prediction_loss_weight=0.1):
    """

    Args:
        llm: a loaded LLM
        medical_hgt: an initialized MedicalHGT
        split_loaders: a dict {train: train_batches_list, val: val_batches_list, test: test_batches_list}
        device: 'cude' if available, else 'cpu'
        file_name: used for saving the model during anf after training
        qa_dataset: the loaded MedMCQA dataset
        prime_kg: nx graph object a subset of PrimKG
        llm_feedbacks_dict: a mapping from questions in the MedMCQA dataset to the pre-computed LLM Feedback, answering the questions with and without context
        question_to_subgraphs_mapping: a mapping from questions in the MedMCQA dataset to their corresponding heterogeneous graphs' nodes (in for of tuples (node_type, node_uid)
        num_epochs: upper bound for the number of epochs
        lr: learning rate
        link_prediction_loss_weight: the weight of the link prediction performance to the performance of the model

    Returns:
        medical_hgt_result: a ModelResult object

    """

    medical_hgt = medical_hgt.to(device)

    medical_hgt.train()

    opt = torch.optim.Adam(medical_hgt.parameters(), lr=lr)

    precision = BinaryPrecision()
    recall = BinaryRecall()
    F1 = BinaryF1Score()

    eval_qa_dataset = pd.DataFrame(qa_dataset['validation'])

    start_time = get_time()
    print(f'Saving results to {file_name}')

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

        print("Train Batches...")
        for batch in tqdm(train_loader):
            batch = batch.to(device)

            opt.zero_grad()

            # HGT forward pass
            pos_train_pred, neg_train_pred, z_dict = medical_hgt(batch)

            pos_train_y = batch["question", "question_correct_answer", "answer"].edge_label.squeeze()
            neg_train_y = batch["question", "question_wrong_answer", "answer"].edge_label.squeeze()

            if pos_train_y.dim() == 0:
                pos_train_y = pos_train_y.view(1)

            if neg_train_y.dim() == 0:
                neg_train_y = neg_train_y.view(1)

            link_prediction_loss = compute_link_prediction_loss(pos_train_pred, neg_train_pred, pos_train_y, neg_train_y, device=device)

            llm_relevancy_loss = compute_llm_relevancy_loss(batch, z_dict, llm_feedbacks_dict)

            # Weighted dual-task loss
            total_loss = link_prediction_loss_weight * link_prediction_loss + llm_relevancy_loss_weight * llm_relevancy_loss

            # Backward pass
            total_loss.backward()
            opt.step()

            # Store results
            pos_y_pred_tensors.append(pos_train_pred.detach())
            neg_y_pred_tensors.append(neg_train_pred.detach())
            pos_y_true_tensors.append(pos_train_y.detach().long())
            neg_y_true_tensors.append(neg_train_y.detach().long())

            train_losses.append(total_loss.detach().item())

        train_end_time = get_time()

        # Accumulate train results
        pos_pred = torch.cat(pos_y_pred_tensors, dim=0).cpu()
        neg_pred = torch.cat(neg_y_pred_tensors, dim=0).cpu()
        pos_true = torch.cat(pos_y_true_tensors, dim=0).cpu()
        neg_true = torch.cat(neg_y_true_tensors, dim=0).cpu()

        pred = torch.cat([pos_pred, neg_pred])
        true = torch.cat([pos_true, neg_true])

        # the training ROC AUC is computed using all the predictions (and ground
        # truth labels) made during the entire epoch, across all batches. Note that
        # this is arguably a bit inconsistent with validation below since it doesn't
        # give the medical_hgt a "second try" for earlier batches, for which it couldn't
        # have yet applied anything it learned in later batches.
        train_roc_auc = roc_auc_score(true.numpy(), pred.numpy)
        train_precision = precision(pred, true)
        train_recall = recall(pred, true)
        train_f1 = F1(pred, true)

        # The validation ROC AUC is computed by running through the validation set
        # at the end of every epoch.
        val_pred, val_true, val_llm_acc_dict = evaluate(llm, medical_hgt, split_loaders, 'val', device, eval_qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping)

        val_roc_auc = roc_auc_score(val_true.numpy(), val_pred.numpy)
        val_precision = precision(val_pred, val_true)
        val_recall = recall(val_pred, val_true)
        val_f1 = F1(val_pred, val_true)

        epoch_result = EpochResult(
            epoch_num=epoch_num,
            train_start_time=train_start_time,
            train_end_time=train_end_time,
            mean_train_loss=round(np.mean(train_losses), 4),
            train_roc_aoc=train_roc_auc,
            train_precision=train_precision,
            train_recall=train_recall,
            train_f1=train_f1,
            val_roc_aoc=val_roc_auc,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            llm_results=val_llm_acc_dict
        )

        epoch_results.append(epoch_result)
        print(f'\r{epoch_result}')

    state_dict = copy.deepcopy(medical_hgt.state_dict())

    # Run through the test set
    test_pred, test_true, test_llm_acc_dict = evaluate(llm, medical_hgt, split_loaders, 'test', device, eval_qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping)

    test_roc_auc = roc_auc_score(test_true.numpy(), test_pred.numpy)
    test_precision = precision(test_pred, test_true)
    test_recall = recall(test_pred, test_true)
    test_f1 = F1(test_pred, test_true)
    medical_hgt.eval()

    end_time = get_time()

    medical_hgt_result = ModelResult(start_time, end_time, epoch_results, state_dict, test_roc_auc, test_precision, test_recall, test_f1, test_llm_acc_dict)
    torch.save(medical_hgt_result, file_name)

    train_time_min = medical_hgt_result.get_total_train_time_min()
    print(f'\rTest Accuracy: {test_roc_auc:.3f}; LLM Results: {test_llm_acc_dict}, Total Train Time: {train_time_min} min')

    return medical_hgt_result


def evaluate(llm, medical_hgt, split_loaders, split_name, device, qa_dataset, prime_kg, llm_feedbacks_dict, question_to_subgraphs_mapping, frac=1.0):
    """

    Args:
        llm: a loaded LLM
        medical_hgt: an initialized MedicalHGT
        split_loaders: a dict {train: train_batches_list, val: val_batches_list, test: test_batches_list}
        split_name: 'val' or 'test'
        device: 'cude' if available, else 'cpu'
        qa_dataset: the loaded MedMCQA dataset
        prime_kg: nx graph object a subset of PrimKG
        llm_feedbacks_dict: a mapping from questions in the MedMCQA dataset to the pre-computed LLM Feedback, answering the questions with and without context
        question_to_subgraphs_mapping: a mapping from questions in the MedMCQA dataset to their corresponding heterogeneous graphs' nodes (in for of tuples (node_type, node_uid)
        frac: a fraction of the batches to process

    Returns:
        pred: link prediction results
        true: link prediction ground truths
        llm_results: llm vanilla and context accuracies (dict)

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

            # Forward pass
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

            # Retrieve the HGT's nodes representations and use them to create context for the validation questions
            correct_answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            answer_letter_to_op_map = {'A': 'opa', 'B': 'opb', 'C': 'opc', 'D': 'opd'}

            vanilla_accuracy_list, vanilla_confidence_list, context_accuracy_list, context_confidence_list = [], [], [], []
            unseen_questions_indices = batch["question", "question_correct_answer", "answer"].edge_label_index[0]
            if unseen_questions_indices.dim() == 0:
                unseen_questions_indices = unseen_questions_indices.unsqueeze(-1)

            for question_index in unseen_questions_indices:

                question_node_representation = torch.index_select(z_dict['question'], 0, question_index)  # z_dict['question'][question_index]

                question_uid = batch['question'].node_uid[question_index].item()
                if question_uid not in llm_feedbacks_dict:
                    continue

                llm_feedback_without_context = llm_feedbacks_dict[question_uid]
                subgraph_tuples = question_to_subgraphs_mapping[question_uid]
                most_relevant_nodes = find_most_relevant_nodes(batch, z_dict, question_node_representation, subgraph_tuples, prime_kg)
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

                # Accumulate Results
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

    pos_pred = torch.cat(pos_y_pred_tensors, dim=0).cpu()
    neg_pred = torch.cat(neg_y_pred_tensors, dim=0).cpu()
    pos_true = torch.cat(pos_y_true_tensors, dim=0).cpu()
    neg_true = torch.cat(neg_y_true_tensors, dim=0).cpu()

    pred = torch.cat([pos_pred, neg_pred])
    true = torch.cat([pos_true, neg_true])

    llm_results = {
        'vanilla_accuracy': sum(average_llm_vanilla_accuracy_list) / max(1, len(average_llm_vanilla_accuracy_list)),
        'context_accuracy': sum(average_llm_context_accuracy_list) / max(1, len(average_llm_context_accuracy_list)),
    }

    return pred, true, llm_results
