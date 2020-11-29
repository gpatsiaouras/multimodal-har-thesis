import torch


def get_accuracy(data_loader, model, device):
    num_correct = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        labels = labels.to(device=device)

        # forward
        scores = model(data.float())
        pred_max_val, pred_max_id = scores.max(1)
        num_correct += int((labels.argmax(1) == pred_max_id).sum())
        num_samples += len(pred_max_id)

    return float(num_correct) / float(num_samples) * 100


@torch.no_grad()
def get_confusion_matrix(data_loader, model):
    all_predictions = torch.tensor([])
    all_labels = torch.tensor([])

    for (data, labels) in data_loader:
        scores = model(data.float())
        all_predictions = torch.cat((all_predictions, scores), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    num_classes = all_labels.shape[1]
    stacked = torch.stack((all_labels.argmax(dim=1), all_predictions.argmax(dim=1)), dim=1)
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p in stacked:
        true_label, predicted_label = p.tolist()
        cmt[true_label, predicted_label] = cmt[true_label, predicted_label] + 1

    return cmt
