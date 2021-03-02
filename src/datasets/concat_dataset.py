from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.labels = datasets[0].labels

    def __getitem__(self, i):
        all_items = tuple(d[i][0] for d in self.datasets)
        # From one of the datasets (doesn't matter which e.g 0) get the current item i
        # and the label which is the second in the tuple so 1
        all_labels = self.datasets[0][i][1]
        return all_items, all_labels

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def get_class_names(self):
        return self.datasets[0].get_class_names()
