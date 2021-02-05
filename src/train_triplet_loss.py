import time
import argparse
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import UtdMhadDataset, BalancedSampler
from losses import TripletLossHard
from models import CNN1D
from tools import train_triplet_loss, get_predictions_with_knn
from transforms import Compose, Normalize, Sampler, Flatten, Jittering
from visualizers import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--mr', type=float, default=None)
args = parser.parse_args()

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

batch_size = 32
num_epochs = 5

mean = [-0.62575306, -0.26179606, -0.07613295, 3.70461374, -4.34395205, -0.09911604]
std = [0.6440941, 0.46361165, 0.43402348, 87.2470291, 100.86503743, 107.77852571]

modality = 'inertial'
actions = [*range(27)]

train_dataset = UtdMhadDataset(modality='inertial', actions=actions, subjects=[1, 3, 5, 7], transform=Compose([
    Normalize(mean, std),
    Jittering([0, 500, 1000]),
    Sampler(107),
    Flatten(),
]))
train_loader = DataLoader(dataset=train_dataset, batch_sampler=BalancedSampler(
    dataset=train_dataset,
    n_classes=len(actions),
    n_samples=4,
    sampler=torch.utils.data.sampler.Sampler(train_dataset),
    batch_size=32,
    drop_last=False
))
val_dataset = UtdMhadDataset(modality='inertial', actions=actions, subjects=[2, 4], transform=Compose([
    Normalize(mean, std),
    Sampler(107),
    Flatten(),
]))
val_loader = DataLoader(val_dataset, batch_size, True)
test_dataset = UtdMhadDataset(modality='inertial', actions=actions, subjects=[6, 8], transform=Compose([
    Normalize(mean, std),
    Sampler(107),
    Flatten(),
]))
test_loader = DataLoader(test_dataset, batch_size, True)

model = CNN1D(len_seq=107 * 6, out_size=128, norm_out=True)
model.to(device)

margin = args.mr
n_neighbors = 4
criterion = TripletLossHard(margin)
lr = args.lr
optimizer = RMSprop(model.parameters(), lr=lr)

experiment = 'MarLrExp_%s_%s_TL_A%s_M%s_LR%s_hard_100ep' % (
    model.name, modality, str(len(actions)), str(margin), str(lr))
print('Experiment:  %s' % experiment)
writer = SummaryWriter('../logs/' + experiment)

min_val_loss, max_val_acc, last_step = train_triplet_loss(model=model,
                                                          criterion=criterion,
                                                          optimizer=optimizer,
                                                          class_names=train_dataset.get_class_names(),
                                                          train_loader=train_loader,
                                                          val_loader=val_loader,
                                                          num_epochs=num_epochs,
                                                          device=device,
                                                          experiment=experiment,
                                                          writer=writer,
                                                          n_neighbors=n_neighbors
                                                          )

cm, test_accuracy = get_predictions_with_knn(
    n_neighbors=n_neighbors,
    train_loader=train_loader,
    test_loader=test_loader,
    model=model,
    device=device
)

print('Test accuracy: %f' % test_accuracy)
cm_image = plot_confusion_matrix(
    cm=cm,
    title='Confusion Matrix- Test Loader',
    normalize=False,
    save=False,
    show_figure=False,
    classes=test_dataset.get_class_names()
)
writer.add_images('ConfusionMatrix/Test', cm_image, dataformats='CHW', global_step=last_step)
writer.add_hparams({'learning_rate': lr, 'margin': margin}, {'val_accuracy': max_val_acc, 'val_loss': min_val_loss})
writer.flush()
writer.close()
