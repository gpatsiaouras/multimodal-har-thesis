import torch
import yaml, os, sys
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from datasets import UtdMhadDataset
from models import CNN1D
from tools import get_accuracy, get_confusion_matrix, load_yaml
from transforms import Sampler, Compose, Flatten, FilterDimensions, Jittering
from configurators import UtdMhadDatasetConfig
from visualizers import plot_accuracy, plot_loss, plot_confusion_matrix, print_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed number generator
torch.manual_seed(0)

# Load parameters from yaml file
yaml_file = sys.argv[1] if len(sys.argv) == 2 else 'parameters/inertial/default.yaml'
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
criterion = functional.mse_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Statistics
train_accuracies = []
test_accuracies = []
losses = []

# Train Network
for epoch in range(num_epochs):
    running_loss = .0
    num_correct = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        labels = labels.to(device=device)

        # forward
        scores = model(data.float())
        loss = criterion(scores, labels.float())

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
    train_loss = running_loss / len(train_loader)
    train_acc = float(num_correct) / float(num_samples) * 100
    test_acc = get_accuracy(test_loader, model, device)
    losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    print('Epoch %d loss: %.3f accuracy: %f, test_accuracy: %f' % (epoch + 1, train_loss, train_acc, test_acc))

# plot results
plot_accuracy(train_acc=train_accuracies, test_acc=test_accuracies)
plot_loss(losses)
plot_confusion_matrix(
    cm=get_confusion_matrix(test_loader, model),
    classes=UtdMhadDatasetConfig().get_class_names()
)
