import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset
from tools import train
from models import BiGRU
from transforms import Resize, Compose, ToSequence, FilterJoints, Normalize, SwapJoints
from visualizers import print_table

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed number generator
torch.manual_seed(0)

# Assign hyper parameters
num_classes = 27
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Data related
# head, left elbow, left hand, right elbow, right hand, left knee, left foot, right knee and right foot
selected_joints = [0, 5, 7, 9, 11, 13, 15, 17, 19]
# Resize (and add padding) to exactly 125 frames
num_frames = 41

# Model params
hidden_size = 128
num_layers = 2
input_size = len(selected_joints)  # nine joints
sequence_length = num_frames * 3  # features(frames) x 3 dimensions xyz per frame

# Print parameters
print_table({
    'num_classes': num_classes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'num_frames': num_frames,
    'input_size': input_size,
    'sequence_length': sequence_length,
})

# Load Data
train_dataset = UtdMhadDataset(modality='skeleton', train=True, transform=Compose([
    Normalize((0, 2)),
    Resize(num_frames),
    FilterJoints(selected_joints),
    # SwapJoints(),
    ToSequence(sequence_length, input_size)
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataset = UtdMhadDataset(modality='skeleton', train=False, transform=Compose([
    Normalize((0, 2)),
    Resize(num_frames),
    FilterJoints(selected_joints),
    # SwapJoints(),
    ToSequence(sequence_length, input_size)
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Model
model = BiGRU(batch_size, input_size, hidden_size, num_layers, num_classes, device).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)


train(model, criterion, optimizer, train_loader, test_loader, num_epochs, batch_size, device)
