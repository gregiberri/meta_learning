import torch


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        datas = [d[i] for d in self.datasets]
        return datas

    def __len__(self):
        return min(len(d) for d in self.datasets)