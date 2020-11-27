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
