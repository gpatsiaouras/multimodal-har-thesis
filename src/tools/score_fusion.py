import torch
from .training import get_predictions, get_num_correct_predictions


def get_score_fusion_accuracy(data_loaders, models, device):
    """
    Receives two lists of data loaders and models (synchronized), gets predictions
    and fuses the data of all the models based on a max, product and sum rule. Returns the accuracy for
    all three rules.
    :param data_loaders: List of data loaders
    :param models: List of models
    :param device: Device to be used
    :return: max rule accuracy, product rule accuracy, sum rule accuracy
    """
    # TODO change hardcoded numbers
    c_scores = torch.zeros((len(models), 416, 27))
    c_labels = torch.zeros((len(models), 416, 27))

    for idx in range(len(models)):
        all_predictions, all_labels = get_predictions(data_loaders[idx], models[idx], device)

        c_scores[idx, :, :] = all_predictions
        c_labels[idx, :, :] = all_labels

    # Perform all three available rules of fusion
    (fused_scores_max, _) = c_scores.max(dim=0)
    fused_scores_prod = c_scores.prod(dim=0)
    fused_scores_sum = c_scores.sum(dim=0)

    # Calculate individual predictions for each rule
    correct_pred_max = get_num_correct_predictions(fused_scores_max, c_labels[0])
    correct_pred_prod = get_num_correct_predictions(fused_scores_prod, c_labels[0])
    correct_pred_sum = get_num_correct_predictions(fused_scores_sum, c_labels[0])

    # Return a tuple of max, product and sum rules accuracy
    return correct_pred_max / 416, correct_pred_prod / 416, correct_pred_sum / 416
