import os
import time

import torch
import torch.nn.functional as functional
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from .model_tools import save_model


def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs, batch_size, device, writer=None):
    """
    Performs training process based on the configuration provided as input. Automatically saves accuracy, loss and
    confusion matrix plots.
    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data
    :param validation_loader: Validation data
    :param num_epochs: Number of epochs to train
    :param batch_size: Batch size
    :param device: Device
    :param writer: SummaryWriter for tensorboard
    """
    # Statistics
    start_time = time.time()
    time_per_epoch = []
    train_accuracies = []
    validation_accuracies = []
    losses = []
    saved_model_path = None

    # Train Network
    for epoch in range(num_epochs):
        running_loss = .0
        num_correct = 0
        num_samples = 0
        epoch_start_time = time.time()

        # Iterate mini batches
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Turn inference mode off (in case)
            model.train()
            # Get data to cuda if possible
            data = data.to(device)
            labels = labels.to(device)

            # forward
            scores = model(data)

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
        # Validation accuracy
        validation_acc = get_accuracy(validation_loader, model, device) * 100
        # Save the model if the validation accuracy is the highest
        if len(validation_accuracies) > 0 and validation_acc > max(validation_accuracies):
            if saved_model_path is not None:
                os.remove(saved_model_path)
            saved_model_path = save_model(model, 'ep%d_bs%d.pt' % (num_epochs, batch_size))
        validation_accuracies.append(validation_acc)
        # Timing
        total_epoch_time = time.time() - epoch_start_time
        time_per_epoch.append(total_epoch_time)
        total_time = time.time() - start_time
        avg_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
        remaining_time = (num_epochs - epoch) * avg_time_per_epoch

        # Tensorboard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', validation_acc, epoch)

        # Debug information
        print('\n=== Epoch %d/%d ===' % (epoch + 1, num_epochs))
        print('Loss: %.3f' % train_loss)
        print('Train accuracy: %f' % train_acc)
        print('Validation accuracy: %f' % validation_acc)
        print('Epoch duration: %s' % time.strftime('%H:%M:%S', time.gmtime(total_epoch_time)))
        print('Elapsed / Remaining time: %s/%s' % (
            time.strftime('%H:%M:%S', time.gmtime(total_time)), time.strftime('%H:%M:%S', time.gmtime(remaining_time))))

    print('Maximum validation accuracy achieved: %f' % max(validation_accuracies))

    return train_accuracies, validation_accuracies, losses


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
def get_predictions(data_loader, model, device, skip_last_fc=False):
    """
    Receives a dataloader a model and a device, runs all the batches to get the predictions
    and returns a tensor with all the predictions and corresponding labels for every sample of the dataset
    :param data_loader: Data loader
    :param model: Model
    :param device: Device to be used
    :param skip_last_fc: whether to return the 2048 vector instead of scores
    :return: predictions, labels
    """
    # Set the model to inference mode
    model.eval()

    # Initiate tensors to hold predictions and labels
    all_predictions = torch.tensor([], device=device)
    all_labels = torch.tensor([], device=device)

    for (data, labels) in data_loader:
        data = data.to(device)
        labels = labels.to(device=device)

        out = model(data, skip_last_fc)
        if skip_last_fc:
            scores = out
        else:
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


def get_predictions_with_knn(n_neighbors, train_loader, test_loader, model, device):
    x_train, y_train = get_predictions(train_loader, model, device, skip_last_fc=True)
    x_test, y_test = get_predictions(test_loader, model, device, skip_last_fc=True)
    y_test = y_test.argmax(1)
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    if device.type == 'cuda':
        x_train = x_train.cpu()
        y_train = y_train.cpu()
        x_test = x_test.cpu()
        y_test = y_test.cpu()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_pred = y_pred.argmax(1)
    test_accuracy = int((y_test == torch.Tensor(y_pred)).sum()) / y_test.shape[0]
    cm = confusion_matrix(y_test, y_pred)

    return cm, test_accuracy