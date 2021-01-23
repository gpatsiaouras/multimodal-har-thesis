import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from configurators import UtdMhadDatasetConfig
from datasets import UtdMhadDataset
from transforms import Compose, Sampler, FilterDimensions

classes = UtdMhadDatasetConfig().get_class_names()

train_dataset = UtdMhadDataset(modality='inertial', train=True, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2])
]))

num_samples = 250

(example, _) = train_dataset[0]
x = np.zeros((num_samples, example.flatten().shape[0]))
y = np.zeros((num_samples,), dtype=int)

for idx in range(num_samples):
    data, labels = train_dataset[idx]
    x[idx, :] = data.flatten()
    y[idx] = labels.argmax()

df = pd.DataFrame(x)
df['y'] = y
df['label'] = df['y'].apply(lambda i: classes[i])

tsne = TSNE(perplexity=10, n_iter=5000, verbose=1)
tsne_features = tsne.fit_transform(x)

df['tsne-2d-one'] = tsne_features[:, 0]
df['tsne-2d-two'] = tsne_features[:, 1]

plt.figure(figsize=(16, 10))

sns.relplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue="y",
    data=df,
    palette=sns.color_palette("dark", np.unique(y).shape[0]),
    # legend="full",
)

plt.show()
