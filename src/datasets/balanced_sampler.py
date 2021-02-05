from torch.utils.data import BatchSampler, Sampler
import numpy as np


class BalancedSampler(BatchSampler):
    """ Sampler for hard and semi-hard batch sampling
    """
    def __init__(self, dataset, n_classes, n_samples, sampler: Sampler[int], batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = dataset
        self.labels = dataset.labels
        self.unique_labels = np.unique(self.labels)
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.unique_labels, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                class_idx = np.where(self.labels == class_)[0]
                indices.extend(np.random.choice(class_idx, self.n_samples, replace=False))
            indices = np.array(indices)
            np.random.shuffle(indices)
            # for ind in indices:
            #     yield ind
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
