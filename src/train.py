import torch
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader

from src.datasets.utd_mhad_inertial import UtdMhadInertialDataset
from src.models.cnn1d import CNN1D
from src.transforms.inertial_transforms import Sampler, Compose, Flatten, FilterDimensions
from src.visualizers.plotter import plot_accuracy, plot_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 1
num_classes = 27
learning_rate = 0.001
batch_size = 256
num_epochs = 200

# Load Data
train_dataset = UtdMhadInertialDataset(transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Flatten(),
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN1D().to(device)

# Loss and optimizer
criterion = functional.mse_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Statistics
accuracies = []
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

        # accuracy
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
    losses.append(train_loss)
    accuracies.append(train_acc)
    print('Epoch %d loss: %.3f accuracy: %f' % (epoch + 1, train_loss, train_acc))

# plot results
plot_accuracy(accuracies)
plot_loss(losses)
