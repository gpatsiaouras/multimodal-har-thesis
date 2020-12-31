import time

import torch
import torch.nn.functional as functional

from configurators import UtdMhadDatasetConfig
from .model_tools import save_model
from visualizers import plot_confusion_matrix, plot_loss, plot_accuracy


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

            running_loss += loss

        # print statistics
        # Loss
        train_loss = running_loss / len(train_loader)
        losses.append(train_loss)
        # Train accuracy
        train_acc = float(num_correct) / float(num_samples) * 100
        train_accuracies.append(train_acc)
        # Test accuracy
        test_acc = get_accuracy(test_loader, model, device)
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


@torch.no_grad()
def get_accuracy(data_loader, model, device):
    """
    Calculates the accuracy of the model given (inference mode) applied in the dataset of the data_loader
    :param data_loader: Data loader
    :param model: Model to be used for predictions
    :param device: Device to be used
    :return: accuracy percentage %
    """
    # Set the model to inference mode
    model.eval()
    num_correct = 0
    num_samples = 0

    # Preparing steps for RNN type of networks
    h = None
    c = None
    if model.name is 'gru':
        h = model.init_hidden()
    elif model.name is 'lstm':
        h, c = model.init_hidden()

    for batch_idx, (data, labels) in enumerate(data_loader):
        # Get data to cuda if possible
        data = data.float().to(device=device)
        labels = labels.to(device=device)

        # forward
        out, _, _ = _forward(model, data, h, c)

        scores = functional.softmax(out, 1)
        pred_max_val, pred_max_id = scores.max(1)
        num_correct += int((labels.argmax(1) == pred_max_id).sum())
        num_samples += len(pred_max_id)

    return float(num_correct) / float(num_samples) * 100


@torch.no_grad()
def get_confusion_matrix(data_loader, model, device):
    """
    Retrieves predictions for the whole dataset, calculates and returns the confusion matrix
    :param data_loader: Data loader
    :param model: Model to be used for predictions
    :param device: Device to be used
    :return: confusion matrix
    """
    # Set the model to inference mode
    model.eval()
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

    num_classes = all_labels.shape[1]
    stacked = torch.stack((all_labels.argmax(dim=1), all_predictions.argmax(dim=1)), dim=1)
    cmt = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p in stacked:
        true_label, predicted_label = p.tolist()
        cmt[true_label, predicted_label] = cmt[true_label, predicted_label] + 1

    return cmt


def _forward(model, data, h, c):
    if model.name is 'gru':
        out, h = model(data, h.data)
        return out, h, None
    elif model.name is 'lstm':
        out, h, c = model(data, h.data, c.data)
        return out, h, c
    else:
        out = model(data)
        return out, None, None
