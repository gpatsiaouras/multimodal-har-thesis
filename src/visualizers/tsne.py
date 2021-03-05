import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def tsne(embeddings, labels, class_names, perplexity=10, n_iter=300, verbose=1):
    """
    Run TSNE and plot the results for a list of embeddings and labels
    :param embeddings:
    :param labels:
    :param class_names:
    :param perplexity:
    :param n_iter:
    :param verbose:
    :return:
    """
    df = pd.DataFrame(embeddings)
    df['y'] = labels
    df['label'] = df['y'].apply(lambda i: class_names[i])

    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, verbose=verbose)
    tsne_features = tsne.fit_transform(embeddings)

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
