import sys

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose

from datasets import UtdMhadDataset, UtdMhadDatasetConfig
from tools import get_accuracy, get_confusion_matrix
# Check that saved model was given
from visualizers import plot_confusion_matrix

if len(sys.argv) < 2:
    print('Not saved model was specified. Exiting...')
    sys.exit(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load parameters
batch_size = 32
num_classes = 27

# Load test dataset
test_dataset = UtdMhadDataset(modality='sdfdi', train=False, transform=Compose([
    Resize(224),
    ToTensor()
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Reinitialise model, and load weights from file
model = models.mobilenet_v2(pretrained=True)
model.name = 'mobilenet_v2'
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(model.last_channel, 2048),
    nn.Linear(2048, num_classes)
)

model.to(device)
model.load_state_dict(torch.load(sys.argv[1]))

print('Test Accuracy: %f' % get_accuracy(test_loader, model, device))
plot_confusion_matrix(
    cm=get_confusion_matrix(test_loader, model, device),
    title='Confusion Matrix - Percentage %',
    normalize=True,
    save=False,
    classes=UtdMhadDatasetConfig().get_class_names()
)
