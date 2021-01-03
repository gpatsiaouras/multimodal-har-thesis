import sys

import torch
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset, UtdMhadDatasetConfig
from models import BiGRU
from tools import get_accuracy, get_confusion_matrix
# Check that saved model was given
from transforms import Normalize, Resize, FilterJoints, ToSequence, Compose
from visualizers import plot_confusion_matrix

if len(sys.argv) < 2:
    print('Not saved model was specified. Exiting...')
    sys.exit(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load parameters
hidden_size = 128
num_layers = 2
batch_size = 32
num_classes = 27
num_frames = 41
selected_joints = [0, 5, 7, 9, 11, 13, 15, 17, 19]
input_size = len(selected_joints)  # number of joints as input size
sequence_length = num_frames * 3  # features(frames) x 3 dimensions xyz per frame

# Load test dataset
test_dataset = UtdMhadDataset(modality='skeleton', train=False, transform=Compose([
    Normalize((0, 2)),
    Resize(num_frames),
    FilterJoints(selected_joints),
    ToSequence(sequence_length, input_size)
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Reinitialise model, and load weights from file
model = BiGRU(batch_size, input_size, hidden_size, num_layers, num_classes, device).to(device)
model.load_state_dict(torch.load(sys.argv[1]))

print('Test Accuracy: %d' % get_accuracy(test_loader, model, device))
plot_confusion_matrix(
    cm=get_confusion_matrix(test_loader, model, device),
    title='Confusion Matrix - Percentage %',
    normalize=True,
    save=False,
    classes=UtdMhadDatasetConfig().get_class_names()
)
