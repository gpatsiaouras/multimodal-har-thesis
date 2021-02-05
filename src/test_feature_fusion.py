import sys

import torch
import torchvision
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset
from models import ELM, CNN1D, MobileNetV2, BiLSTM
from tools import get_fused_feature_vector, get_predictions
from transforms import Sampler, FilterDimensions, Flatten, Compose, Normalize, Resize, FilterJoints

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
batch_size = 64
num_classes = 27
input_size = 2048
hidden_size = 8192
# Skeleton stuff
num_frames = 41
hidden_size_skeleton = 128
num_layers = 2
selected_joints = [0, 5, 7, 9, 11, 13, 15, 17, 19]
input_size_skeleton = len(selected_joints) * 3  # input size is joints times axis (per joint)
sequence_length = num_frames

inertial_mean = [-0.62575306, -0.26179606, -0.07613295, 3.70461374, -4.34395205, -0.09911604]
inertial_std = [0.6440941, 0.46361165, 0.43402348, 87.2470291, 100.86503743, 107.77852571]
skeleton_mean = [-0.09214367, -0.29444627, 2.87122181]
skeleton_std = [0.13432376, 0.46162172, 0.12374677]

# Datasets initialization train/test
train_dataset_inertial = UtdMhadDataset(modality='inertial', subjects=[1, 3, 5, 7], transform=Compose([
    Normalize(inertial_mean, inertial_std),
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Flatten(),
]))
test_dataset_inertial = UtdMhadDataset(modality='inertial', subjects=[6, 8], transform=Compose([
    Normalize(inertial_mean, inertial_std),
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Flatten(),
]))
train_dataset_rgb = UtdMhadDataset(modality='sdfdi', subjects=[1, 3, 5, 7], transform=Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
]))
test_dataset_rgb = UtdMhadDataset(modality='sdfdi', subjects=[6, 8], transform=Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
]))
train_dataset_skeleton = UtdMhadDataset(modality='skeleton', subjects=[1, 3, 5, 7], transform=Compose([
    Normalize(skeleton_mean, skeleton_std),
    Resize(num_frames),
    FilterJoints(selected_joints),
    Flatten(),
]))
test_dataset_skeleton = UtdMhadDataset(modality='skeleton', subjects=[6, 8], transform=Compose([
    Normalize(skeleton_mean, skeleton_std),
    Resize(num_frames),
    FilterJoints(selected_joints),
    Flatten(),
]))

# Loaders
train_loader_inertial = DataLoader(dataset=train_dataset_inertial, batch_size=batch_size)
test_loader_inertial = DataLoader(dataset=test_dataset_inertial, batch_size=batch_size)
train_loader_rgb = DataLoader(dataset=train_dataset_rgb, batch_size=batch_size)
test_loader_rgb = DataLoader(dataset=test_dataset_rgb, batch_size=batch_size)
train_loader_skeleton = DataLoader(dataset=train_dataset_skeleton, batch_size=batch_size)
test_loader_skeleton = DataLoader(dataset=test_dataset_skeleton, batch_size=batch_size)

# Initialize models and load saved weights
model_inertial = CNN1D(107 * 3, 27).to(device)
model_inertial.load_state_dict(torch.load(sys.argv[1]))
model_rgb = MobileNetV2(num_classes, pretrained=False).to(device)
model_rgb.load_state_dict(torch.load(sys.argv[2]))
model_skeleton = BiLSTM(input_size_skeleton, hidden_size_skeleton, num_layers, num_classes).to(device)
model_skeleton.load_state_dict(torch.load(sys.argv[3]))

# Get predictions from each model for each loader
print('Retrieving the 2048 feature vectors for all models...')
train_data_inertial = get_predictions(train_loader_inertial, model_inertial, device, apply_softmax=False)
test_data_inertial = get_predictions(test_loader_inertial, model_inertial, device, apply_softmax=False)
print('Inertial.. OK')
train_data_rgb = get_predictions(train_loader_rgb, model_rgb, device, apply_softmax=False)
test_data_rgb = get_predictions(test_loader_rgb, model_rgb, device, apply_softmax=False)
print('RGB.. OK')
train_data_skeleton = get_predictions(train_loader_skeleton, model_skeleton, device, apply_softmax=False)
test_data_skeleton = get_predictions(test_loader_skeleton, model_skeleton, device, apply_softmax=False)
print('Skeleton.. OK')

# Get fused data from all models
print('Performing feature fusion...')
elm_train_data = get_fused_feature_vector(device, train_data_inertial, train_data_rgb)
elm_test_data = get_fused_feature_vector(device, test_data_inertial, test_data_rgb)

# Elm initialization
elm = ELM(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size, device=device)

# Prepare data and labels, x and y
(elm_train_data_data, elm_train_data_labels) = elm_train_data
(elm_test_data_data, elm_test_data_labels) = elm_test_data

# Fit the ELM in training data. For labels use any of the three, they are all the same since shuffle is off.
print('Training elm network...')
elm.fit(elm_train_data_data, elm_train_data_labels)

# Get accuracy on test data
accuracy = elm.evaluate(elm_test_data_data, elm_test_data_labels)

print('ELM Accuracy: %f' % accuracy)
