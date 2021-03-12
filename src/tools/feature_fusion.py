import torch


def get_fused_scores(all_scores, new_scores, rule):
    if all_scores is None:
        return new_scores

    if rule == 'concat':
        return torch.cat((all_scores, new_scores), dim=1)
    elif rule == 'avg':
        return torch.mean(torch.stack((all_scores, new_scores)), dim=0)
    else:
        raise ValueError(rule + ' is not a valid rule')


def get_fused_labels(all_labels, new_labels):
    if all_labels is None:
        return new_labels

    assert int((all_labels - new_labels).sum()) == 0
    return all_labels
