import sys
import time
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset, UtdMhadDatasetConfig
from models import CNN1D
from tools import get_accuracy, load_yaml, get_confusion_matrix, save_model
from transforms import Sampler, Compose, Flatten, FilterDimensions, Jittering
from visualizers import plot_accuracy, plot_loss, print_table, plot_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed number generator
torch.manual_seed(0)

# Load parameters from yaml file.
if len(sys.argv) == 2:
    yaml_file = sys.argv[1]
else:
    yaml_file = os.path.join(os.path.dirname(__file__), 'parameters/inertial/default.yaml')
param_config = load_yaml(yaml_file)

# Assign hyper parameters
num_classes = param_config.get('dataset').get('num_classes')
learning_rate = param_config.get('hyper_parameters').get('learning_rate')
batch_size = param_config.get('hyper_parameters').get('batch_size')
num_epochs = param_config.get('hyper_parameters').get('num_epochs')
jitter_factor = param_config.get('hyper_parameters').get('jitter_factor')

# Print parameters
print_table({
    'num_classes': num_classes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'jitter_factor': jitter_factor
})

# Load Data
train_dataset = UtdMhadDataset(modality='inertial', train=True, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Jittering(jitter_factor),
    Flatten(),
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = UtdMhadDataset(modality='inertial', train=False, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Jittering(jitter_factor),
    Flatten(),
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN1D().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)

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
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Turn inference mode off (in case)
        model.train()
        # Get data to cuda if possible
        data = data.to(device=device)
        labels = labels.to(device=device)

        # forward
        scores = model(data.float())
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
    print('=== Epoch %d ===' % (epoch + 1))
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
