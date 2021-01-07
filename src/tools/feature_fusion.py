import torch


@torch.no_grad()
def get_fused_feature_vector(device, *args):
    all_feature_vectors = torch.zeros(len(args), args[0][0].shape[0], args[0][0].shape[1], device=device)
    for idx in range(len(args)):
        (data, labels) = args[idx]
        all_feature_vectors[idx] = data

    # Returns the averages feature vectors and labels (it doesn't matter which ones every one of them is the same)
    return torch.mean(all_feature_vectors, dim=0), args[0][1]
