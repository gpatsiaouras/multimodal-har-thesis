import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossHard(nn.Module):
    """
    Batch hard/semi-hard manner of triplet loss for embeddings
    Adapted from: https://github.com/lyakaap/NetVLAD-pytorch/blob/master/hard_triplet_loss.py

    Attributes
    ----------
    margin: float
        a value of margin for triplet loss

    Methods
    -------
    forward(embeddings, labels)
        forward pass

    calculate_triplet_loss(embeddings, labels)
        computes triplet loss in the batch hard manner
    """

    def __init__(self, margin=1.0, semi_hard=False):
        super().__init__()
        self.margin = margin
        self.semi_hard = semi_hard

    def forward(self, embeddings, labels):
        return self.calculate_triplet_loss(embeddings, labels)

    def calculate_triplet_loss(self, embeddings, labels):
        matrix = torch.cdist(embeddings, embeddings, p=2.0)
        not_diag = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask_matrix = labels_equal & not_diag
        if not self.semi_hard:
            hardest_pos = (matrix * pos_mask_matrix).max(1)[0]
            hardest_neg = torch.min(matrix + labels_equal * torch.max(matrix), dim=1)[0]
            loss = torch.mean(F.relu(hardest_pos - hardest_neg + self.margin))
        else:
            neg_mask_matrix = ~labels_equal
            idx = (pos_mask_matrix * matrix).nonzero().split(1, dim=1)
            all_pos_distances = matrix[idx].squeeze()
            neg_mask_semi = neg_mask_matrix[idx[0]].squeeze()
            losses = (-matrix[idx[0]].squeeze() + all_pos_distances.reshape(-1, 1) + self.margin) * neg_mask_semi
            semi_hard_losses = (losses < self.margin).int() * (losses > 0).int() * losses
            loss = semi_hard_losses.sum() / ((semi_hard_losses != 0).sum().int() + 1e-16)
        return loss
