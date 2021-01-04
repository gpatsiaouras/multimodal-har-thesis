import sys

import torch
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset, UtdMhadDatasetConfig
from models import CNN1D
from tools import get_accuracy, get_confusion_matrix
from transforms import Sampler, FilterDimensions, Flatten, Compose, Jittering

# Check that saved model was given
from visualizers import plot_confusion_matrix

if len(sys.argv) < 2:
    print('Not saved model was specified. Exiting...')
    sys.exit(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load parameters
batch_size = 16

# Load test dataset
test_dataset = UtdMhadDataset(modality='inertial', train=False, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Flatten(),
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Reinitialise model, and load weights from file
model = CNN1D().to(device)
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

print('Test Accuracy: %f' % get_accuracy(test_loader, model, device))
plot_confusion_matrix(
    cm=get_confusion_matrix(test_loader, model, device),
    title='Confusion Matrix - Percentage %',
    normalize=True,
    save=False,
    classes=UtdMhadDatasetConfig().get_class_names()
)
