import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def run_tsne(embeddings, labels, class_names, perplexity=10, n_iter=300, verbose=1, filename='default.png', save=False,
             show=True):
    """
    Run TSNE and plot the results for a list of embeddings and labels
    :param embeddings:
    :param labels:
    :param class_names:
    :param perplexity:
    :param n_iter:
    :param verbose:
    :param filename:
    :param save:
    :param show:
    :return:
    """
    df = pd.DataFrame(embeddings)
    df['y'] = labels
    df['label'] = df['y'].apply(lambda i: class_names[i])

    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, verbose=verbose)
    tsne_features = tsne.fit_transform(embeddings)

    df['dim-1'] = tsne_features[:, 0]
    df['dim-2'] = tsne_features[:, 1]

    plt.figure(figsize=(16, 10))

    sns.relplot(
        x="dim-1",
        y="dim-2",
        hue="y",
        data=df,
        palette=sns.color_palette("dark", np.unique(labels).shape[0]),
        legend=False,
    )

    if save:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
