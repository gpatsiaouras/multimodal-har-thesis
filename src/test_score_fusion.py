import sys

import torch
import torchvision
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset, UtdMhadDatasetConfig
from models import CNN1D, BiGRU, MobileNetV2
from tools import get_score_fusion_accuracy, get_confusion_matrix_multiple_models, plot_confusion_matrix
from transforms import Compose, Sampler, FilterDimensions, Flatten, Normalize, FilterJoints, ToSequence, Resize

num_classes = 27
batch_size = 32
# Skeleton stuff
num_frames = 41
hidden_size = 128
num_layers = 2
selected_joints = [0, 5, 7, 9, 11, 13, 15, 17, 19]
input_size = len(selected_joints)  # number of joints as input size
sequence_length = num_frames * 3  # features(frames) x 3 dimensions xyz per frame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_list = []
data_loaders_list = []

inertial_included = False
rgb_included = False
skeleton_included = False

# Inertial model
if len(sys.argv) > 1:
    model_inertial = CNN1D().to(device)
    model_inertial.load_state_dict(torch.load(sys.argv[1]))
    test_dataset_inertial = UtdMhadDataset(modality='inertial', train=False, transform=Compose([
        Sampler(107),
        FilterDimensions([0, 1, 2]),
        Flatten(),
    ]))
    test_loader_inertial = DataLoader(dataset=test_dataset_inertial, batch_size=batch_size, shuffle=False,
                                      drop_last=True)
    models_list.append(model_inertial)
    data_loaders_list.append(test_loader_inertial)
    inertial_included = True

if len(sys.argv) > 2:
    model_rgb = MobileNetV2(num_classes).to(device)
    model_rgb.load_state_dict(torch.load(sys.argv[2]))

    test_dataset_rgb = UtdMhadDataset(modality='sdfdi', train=False, transform=Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor()
    ]))
    test_loader_rgb = DataLoader(dataset=test_dataset_rgb, batch_size=batch_size, shuffle=False, drop_last=True)
    models_list.append(model_rgb)
    data_loaders_list.append(test_loader_rgb)
    rgb_included = True

if len(sys.argv) > 3:
    model_skeleton = BiGRU(batch_size, input_size, hidden_size, num_layers, num_classes, device).to(device)
    model_skeleton.load_state_dict(torch.load(sys.argv[3]))

    test_dataset_skeleton = UtdMhadDataset(modality='skeleton', train=False, transform=Compose([
        Normalize((0, 2)),
        Resize(num_frames),
        FilterJoints(selected_joints),
        ToSequence(sequence_length, input_size)
    ]))
    test_loader_skeleton = DataLoader(dataset=test_dataset_skeleton, batch_size=batch_size, shuffle=False,
                                      drop_last=True)
    models_list.append(model_skeleton)
    data_loaders_list.append(test_loader_skeleton)
    skeleton_included = True

# Get three different accuracies for each different fusion rule
max_acc, prod_acc, sum_acc = get_score_fusion_accuracy(data_loaders_list, models_list, device)
accuracies = [max_acc, prod_acc, sum_acc]
# Print table of which modality is included and the fusion scores
table = PrettyTable()
table.field_names = ['Modalities', 'Max rule', 'Product rule', 'Sum rule']
modalities_str = '%s%s%s' % (
    'inertial' if inertial_included else '',
    ',rgb' if rgb_included else '',
    ',skeleton' if skeleton_included else '',
)
table.add_row([modalities_str, '%f' % max_acc, '%f' % prod_acc, '%f' % sum_acc])
print(table)

# Plot confusion matrix based on the product rule
plot_confusion_matrix(
    cm=get_confusion_matrix_multiple_models(data_loaders_list, models_list, device),
    title='Confusion Matrix - Percentage %',
    normalize=True,
    save=True,
    classes=UtdMhadDatasetConfig().get_class_names()
)
