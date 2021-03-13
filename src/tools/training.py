import os
import time

import torch
import torch.nn.functional as functional
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from visualizers import plot_confusion_matrix, print_epoch_info
from .model_tools import save_model


def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs, device, experiment, writer=None):
    """
    Performs training process based on the configuration provided as input. Automatically saves accuracy, loss and
    confusion matrix plots.
    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data
    :param validation_loader: Validation data
    :param num_epochs: Number of epochs to train
    :param experiment: Name of the experiment running
    :param device: Device
    :param writer: SummaryWriter for tensorboard
    """
    # Statistics
    start_time = time.time()
    time_per_epoch = []
    train_accuracies = []
    validation_accuracies = []
    train_losses = []
    validation_losses = []
    saved_model_path = None

    # Train Network
    for epoch in range(num_epochs):
        train_running_loss = .0
        num_correct = 0
        num_samples = 0
        epoch_start_time = time.time()

        # Iterate mini batches
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Turn inference mode off (in case)
            model.train()
            # Get data to cuda if possible
            data = data.float().to(device)
            labels = labels.long().to(device)

            # forward
            scores = model(data)

            _, input_indices = torch.max(labels, dim=1)
            loss = criterion(scores, input_indices)

            # train accuracy
            num_correct += get_num_correct_predictions(scores, labels)
            num_samples += len(labels)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            train_running_loss += loss.item()

        ##########
        # Losses #
        ##########
        # Training Loss
        train_loss = train_running_loss / len(train_loader)
        train_losses.append(train_loss)
        # Validation Loss
        validation_loss, _, _ = get_loss(validation_loader, model, device, criterion)

        ##############
        # Accuracies #
        ##############
        # Train accuracy
        train_acc = float(num_correct) / float(num_samples)
        train_accuracies.append(train_acc)
        # Validation accuracy
        validation_acc = get_accuracy(validation_loader, model, device)
        # Save the model if the validation accuracy is the highest
        if len(validation_accuracies) > 0 and validation_acc > max(validation_accuracies):
            if saved_model_path is not None:
                os.remove(saved_model_path)
            saved_model_path = save_model(model, '%s_best_val.pt' % experiment)
        validation_accuracies.append(validation_acc)
        validation_losses.append(validation_loss)

        # Tensorboard
        if writer:
            writer.add_scalar('Loss/train', train_loss, global_step=epoch)
            writer.add_scalar('Loss/validation', validation_loss, global_step=epoch)
            writer.add_scalar('Accuracy/train', train_acc, global_step=epoch)
            writer.add_scalar('Accuracy/validation', validation_acc, global_step=epoch)

        # Debug information
        print_epoch_info(start_time, epoch_start_time, time_per_epoch, epoch, num_epochs, train_loss, validation_loss,
                         train_acc, validation_acc)

    return train_accuracies, validation_accuracies, train_losses, validation_losses


def add_histograms(writer, model, step, subtitle=''):
    """
    Adds histogram data according to the model to tensorboard.
    :param writer: SummaryWriter
    :param model: Model
    :param step: global step
    :param subtitle: When going deeper provide a name to separate the fc1, fc2, fc3
    :return:
    """
    if hasattr(model, 'fc1'):
        writer.add_histogram(subtitle + 'fc1', model.fc1.weight, global_step=step)
    if hasattr(model, 'fc2'):
        writer.add_histogram(subtitle + 'fc2', model.fc2.weight, global_step=step)
    if hasattr(model, 'fc3'):
        writer.add_histogram(subtitle + 'fc3', model.fc3.weight, global_step=step)
    if hasattr(model, 'last_fc'):
        writer.add_histogram(subtitle + 'last_fc', model.last_fc.weight, global_step=step)
    if hasattr(model, 'mlp'):
        add_histograms(writer, model.mlp, step, subtitle='mlp')
    if hasattr(model, 'n1'):
        add_histograms(writer, model.mlp, step, subtitle='n1')
    if hasattr(model, 'n2'):
        add_histograms(writer, model.mlp, step, subtitle='n2')
    if hasattr(model, 'n3'):
        add_histograms(writer, model.mlp, step, subtitle='n3')


def train_triplet_loss(model, criterion, optimizer, class_names, train_loader, val_loader, num_epochs, device,
                       experiment, n_neighbors, writer, scheduler=None, verbose=False, skip_accuracy=False):
    start_time = time.time()
    time_per_epoch = []
    saved_model_path = None
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_accuracy = -1
    val_accuracy = -1
    scores_concat = None
    labels_concat = None

    for epoch in range(num_epochs):
        scores_concat = torch.tensor([], device=device)
        labels_concat = torch.tensor([], device=device)
        train_running_loss = .0
        epoch_start_time = time.time()
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Turn inference mode off (in case)
            model.train()
            # Get data to cuda if possible
            if isinstance(data, list):
                data = tuple(d.float().to(device) for d in data)
            else:
                data = data.float().to(device)
            targets = targets.to(device)

            # Train scores
            scores = model(data)
            labels = targets.argmax(dim=1)
            loss = criterion(scores, labels)
            train_running_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            scores_concat = torch.cat((scores_concat, scores), dim=0)
            labels_concat = torch.cat((labels_concat, labels), dim=0)

        add_histograms(writer, model, epoch)
        writer.add_scalar('Embeddings/distances', torch.sum(torch.cdist(scores_concat, scores_concat)),
                          global_step=epoch)
        # Calculations
        train_loss = train_running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, global_step=epoch)
        val_loss, val_scores, _ = get_loss(val_loader, model, device, criterion)
        if len(val_losses) > 0:
            # After the first save only for better result in validation
            if val_loss < min(val_losses):
                os.remove(saved_model_path)
                saved_model_path = save_model(model, '%s_best_val.pt' % experiment)
        else:
            # Save the model the first time without checking that the validation was reduced.
            print('Saved model for the first time')
            saved_model_path = save_model(model, '%s_initial.pt' % experiment)
        val_losses.append(val_loss)
        if not skip_accuracy:
            writer.add_scalar('Embeddings/distances_val', torch.sum(torch.cdist(val_scores, val_scores)),
                              global_step=epoch)
            writer.add_scalar('Loss/validation', val_loss, global_step=epoch)
            if scheduler:
                scheduler.step(val_loss)
            # Confusion in general
            val_cm, val_accuracy, _, _ = get_predictions_with_knn(n_neighbors, train_loader, val_loader, model, device)
            train_cm, train_accuracy, _, _ = get_predictions_with_knn(n_neighbors, train_loader, train_loader, model,
                                                                      device)

            val_accuracies.append(val_accuracy)
            train_accuracies.append(train_accuracy)
            writer.add_scalar('Accuracy/validation', val_accuracy, global_step=epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, global_step=epoch)
            train_image = plot_confusion_matrix(
                cm=train_cm,
                title='Confusion Matrix - Percentage % - Train Loader',
                normalize=False,
                save=False,
                show_figure=False,
                classes=class_names
            )
            val_image = plot_confusion_matrix(
                cm=val_cm,
                title='Confusion Matrix - Percentage % - Val Loader',
                normalize=False,
                save=False,
                show_figure=False,
                classes=class_names
            )
            writer.add_images('ConfusionMatrix/Train', train_image, dataformats='CHW', global_step=epoch)
            writer.add_images('ConfusionMatrix/Validation', val_image, dataformats='CHW', global_step=epoch)

        if verbose:
            # Timing
            total_epoch_time = time.time() - epoch_start_time
            time_per_epoch.append(total_epoch_time)

            # Debug information
            print_epoch_info(start_time, epoch_start_time, time_per_epoch, epoch, num_epochs, train_loss, val_loss,
                             train_accuracy, val_accuracy)

    writer.add_embedding(scores_concat, metadata=[class_names[idx] for idx in labels_concat.int().tolist()],
                         tag="train (%f%%)" % train_accuracy)
    return train_losses, val_losses, val_accuracies, train_accuracies


def train_simple(model, criterion, optimizer, epochs, data, labels):
    model.train()
    for epoch in range(epochs):
        scores = model(data)
        loss = criterion(scores, labels.argmax(1))
        accu = get_num_correct_predictions(scores, labels) / labels.shape[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch %d/%d: loss %.2f accu %.2f' % (epoch, epochs, loss.item(), accu))


@torch.no_grad()
def get_loss(data_loader, model, device, criterion):
    """
    Get inference loss of data_loader
    :param data_loader: DataLoader
    :param model: Model
    :param device: Device (cuda or cpu)
    :param criterion: Loss function
    :return: loss
    """
    model.eval()
    scores, labels = get_predictions(data_loader, model, device, apply_softmax=False)
    loss = criterion(scores, labels.argmax(dim=1))

    return loss.item(), scores, labels


def get_accuracy(data_loader, model, device):
    """
    Calculates the accuracy of the model given (inference mode) applied in the dataset of the data_loader
    :param data_loader: Data loader
    :param model: Model to be used for predictions
    :param device: Device to be used
    :return: accuracy percentage %
    """
    all_predictions, all_labels = get_predictions(data_loader, model, device, apply_softmax=True)

    correct_pred_max = get_num_correct_predictions(all_predictions, all_labels)

    return correct_pred_max / all_predictions.shape[0]


@torch.no_grad()
def get_accuracy_simple(model, data, labels):
    model.eval()
    test_scores = model(data)
    test_scores = functional.softmax(test_scores, 1)

    return get_num_correct_predictions(test_scores, labels) / labels.shape[0]


def get_confusion_matrix(data_loader, model, device):
    """
    Retrieves predictions for the whole dataset, calculates and returns the confusion matrix
    :param data_loader: Data loader
    :param model: Model to be used for predictions
    :param device: Device to be used
    :return: confusion matrix
    """
    all_predictions, all_labels = get_predictions(data_loader, model, device, apply_softmax=True)

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
        all_predictions, all_labels = get_predictions(data_loaders[idx], models[idx], device, apply_softmax=True)

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
def get_predictions(data_loader, model, device, apply_softmax=False):
    """
    Receives a dataloader a model and a device, runs all the batches to get the predictions
    and returns a tensor with all the predictions and corresponding labels for every sample of the dataset
    :param data_loader: Data loader
    :param model: Model
    :param device: Device to be used
    :param apply_softmax: Whether to apply softmax in the end
    :return: predictions, labels
    """
    # Set the model to inference mode
    model.eval()

    # Initiate tensors to hold predictions and labels
    all_predictions = torch.tensor([], device=device)
    all_labels = torch.tensor([], device=device)

    for (data, labels) in data_loader:
        if isinstance(data, list):
            data = tuple(d.float().to(device) for d in data)
        else:
            data = data.float().to(device)
        labels = labels.long().to(device=device)

        out = model(data)
        if apply_softmax:
            scores = functional.softmax(out, 1)
        else:
            scores = out
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
    """
    Receives the train and test loaders and number of neighbors and trains a KNN classifier on the train data, and then
    predicts on the test loader. Returns the confusion matrix, test accuracy, scores and labes
    :param n_neighbors: Number of neighbors for the KNN
    :param train_loader: Train loader
    :param test_loader: Test Loader
    :param model: Model to use
    :param device: Torch device
    :return: confusion matrix, test accuracy, scores, labels
    """
    train_scores, train_labels = get_predictions(train_loader, model, device, apply_softmax=False)
    test_scores, test_labels = get_predictions(test_loader, model, device, apply_softmax=False)
    test_labels = test_labels.argmax(1)
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    if device.type == 'cuda':
        train_scores = train_scores.cpu()
        train_labels = train_labels.cpu()
        test_scores = test_scores.cpu()
        test_labels = test_labels.cpu()
    classifier.fit(train_scores, train_labels.argmax(1))
    test_knn_pred = classifier.predict(test_scores)
    # y_pred = y_pred.argmax(1)
    test_accuracy = int((test_labels == torch.Tensor(test_knn_pred)).sum()) / test_labels.shape[0]
    cm = confusion_matrix(test_labels, test_knn_pred)

    return cm, test_accuracy, test_scores, test_labels
