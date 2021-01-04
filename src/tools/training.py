import time

import torch
import torch.nn.functional as functional

from configurators import UtdMhadDatasetConfig
from visualizers import plot_confusion_matrix, plot_loss, plot_accuracy
from .model_tools import save_model


def train(model, criterion, optimizer, train_loader, test_loader, num_epochs, batch_size, device):
    """
    Performs training process based on the configuration provided as input. Automatically saves accuracy, loss and
    confusion matrix plots.
    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data
    :param test_loader: Test data
    :param num_epochs: Number of epochs to train
    :param batch_size: Batch size
    :param device: Device
    """
    # Statistics
    start_time = time.time()
    train_accuracies = []
    test_accuracies = []
    losses = []

    # Train Network
    for epoch in range(num_epochs):
        running_loss = .0
        num_correct = 0
        num_samples = 0
        epoch_start_time = time.time()

        # Preparing steps for RNN type of networks
        h = None
        c = None
        if model.name is 'gru':
            h = model.init_hidden()
        elif model.name is 'lstm':
            h, c = model.init_hidden()

        # Iterate mini batches
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Turn inference mode off (in case)
            model.train()
            # Get data to cuda if possible
            data = data.to(device).float()
            labels = labels.to(device)

            # forward
            scores, h, c = _forward(model, data, h, c)

            _, input_indices = torch.max(labels, dim=1)
            loss = criterion(scores, input_indices)

            # train accuracy
            pred_max_val, pred_max_id = scores.max(1)
            num_correct += int((labels.argmax(1) == pred_max_id).sum())
            num_samples += len(pred_max_id)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            running_loss += loss.item()

        # print statistics
        # Loss
        train_loss = running_loss / len(train_loader)
        losses.append(train_loss)
        # Train accuracy
        train_acc = float(num_correct) / float(num_samples) * 100
        train_accuracies.append(train_acc)
        # Test accuracy
        test_acc = get_accuracy(test_loader, model, device) * 100
        test_accuracies.append(test_acc)
        # Timing
        total_epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print('\n=== Epoch %d/%d ===' % (epoch + 1, num_epochs))
        print('loss: %.3f' % train_loss)
        print('accuracy: %f' % train_acc)
        print('test_accuracy: %f' % test_acc)
        print('epoch duration: %s' % time.strftime('%H:%M:%S', time.gmtime(total_epoch_time)))
        print('total duration until now: %s' % time.strftime('%H:%M:%S', time.gmtime(total_time)))

    # Save the model after finished training. Add the number of epochs and
    # batch size in the filename for clarity
    save_model(model, 'ep%d_bs%d.pt' % (num_epochs, batch_size))

    # plot results
    plot_accuracy(train_acc=train_accuracies, test_acc=test_accuracies, save=True)
    plot_loss(losses, save=True)
    plot_confusion_matrix(
        cm=get_confusion_matrix(test_loader, model, device),
        title='Confusion Matrix - Percentage %',
        normalize=True,
        save=True,
        classes=UtdMhadDatasetConfig().get_class_names()
    )


def get_accuracy(data_loader, model, device):
    """
    Calculates the accuracy of the model given (inference mode) applied in the dataset of the data_loader
    :param data_loader: Data loader
    :param model: Model to be used for predictions
    :param device: Device to be used
    :return: accuracy percentage %
    """
    all_predictions, all_labels = get_predictions(data_loader, model, device)

    correct_pred_max = get_num_correct_predictions(all_predictions, all_labels)

    return correct_pred_max / all_predictions.shape[0]


def get_confusion_matrix(data_loader, model, device):
    """
    Retrieves predictions for the whole dataset, calculates and returns the confusion matrix
    :param data_loader: Data loader
    :param model: Model to be used for predictions
    :param device: Device to be used
    :return: confusion matrix
    """
    all_predictions, all_labels = get_predictions(data_loader, model, device)

    num_classes = all_labels.shape[1]
    stacked = torch.stack((all_labels.argmax(dim=1), all_predictions.argmax(dim=1)), dim=1)
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p in stacked:
        true_label, predicted_label = p.tolist()
        cmt[true_label, predicted_label] = cmt[true_label, predicted_label] + 1

    return cmt


def get_confusion_matrix_multiple_models(data_loaders, models, device):
    """
    Gets predictions from multiple models, fuses scores based on product rule and calculates and returns
    confusion matrix
    :param data_loaders: List of data loaders
    :param models: List of models
    :param device: Device to be used
    :return: confusion matrix
    """
    # TODO change hardcoded numbers
    c_scores = torch.zeros((len(models), 416, 27))
    c_labels = torch.zeros((len(models), 416, 27))

    for idx in range(len(models)):
        all_predictions, all_labels = get_predictions(data_loaders[idx], models[idx], device)

        c_scores[idx, :, :] = all_predictions
        c_labels[idx, :, :] = all_labels


    fused_scores = c_scores.prod(dim=0)
    num_classes = c_labels[0].shape[1]
    stacked = torch.stack((c_labels[0].argmax(dim=1), fused_scores.argmax(dim=1)), dim=1)
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p in stacked:
        true_label, predicted_label = p.tolist()
        cmt[true_label, predicted_label] = cmt[true_label, predicted_label] + 1

    return cmt


@torch.no_grad()
def get_predictions(data_loader, model, device):
    """
    Receives a dataloader a model and a device, runs all the batches to get the predictions
    and returns a tensor with all the predictions and corresponding labels for every sample of the dataset
    :param data_loader: Data loader
    :param model: Model
    :param device: Device to be used
    :return: predictions, labels
    """
    # Set the model to inference mode
    model.eval()

    # Initiate tensors to hold predictions and labels
    all_predictions = torch.tensor([], device=device)
    all_labels = torch.tensor([], device=device)

    # Preparing steps for RNN type of networks
    h = None
    c = None
    if model.name is 'gru':
        h = model.init_hidden()
    elif model.name is 'lstm':
        h, c = model.init_hidden()

    for (data, labels) in data_loader:
        data = data.float().to(device=device)
        labels = labels.to(device=device)

        out, _, _ = _forward(model, data, h, c)
        scores = functional.softmax(out, 1)
        all_predictions = torch.cat((all_predictions, scores), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    return all_predictions, all_labels


def get_num_correct_predictions(scores, labels):
    """
    Calculates the number of correct prediction given the scores (output of model for every sample) and the labels
    :param scores: tensor
    :param labels: tensor
    :return: number of correct predictions
    """
    return int((labels.argmax(1) == scores.argmax(1)).sum())


def _forward(model, data, h, c):
    """
    Performs a forward pass based on the model given. It automatically handles LSTM, GRU, CNN and other networks.
    :param model: Model to use forward
    :param data: input
    :param h: hidden layer
    :param c: c layer (for lstm)
    :return: scores
    """
    if model.name is 'gru':
        out, h = model(data, h.data)
        return out, h, None
    elif model.name is 'lstm':
        out, h, c = model(data, h.data, c.data)
        return out, h, c
    else:
        out = model(data)
        return out, None, None
