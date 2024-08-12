import sklearn as skl
from sklearn.datasets import make_classification
import torch as pt
from torch.utils import data
import numpy as np
from dataclasses import dataclass


@dataclass()
class Dataset(data.Dataset):
    x: pt.Tensor
    y: pt.Tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __iter__(self):
        return zip(self.x, self.y)

    def __len__(self):
        return self.x.shape[0]

    def __repr__(self):
        return f'{type(self)}({self.x.shape})'


def make_dataset(train: int, test: int, sample: int, positive_ratio: float = 0.5):
    n_samples = train + test + sample

    #generate dataset with n_sameples data and 20 features
    x, y = make_classification(n_samples, weights=[1 - positive_ratio, positive_ratio], random_state=42)
    print(x.shape)
    actual_positive_samples = np.sum(y == 1)
    print(f"Number of positive samples: {actual_positive_samples / n_samples}")

    x = pt.tensor(x, dtype=pt.float)
    #do the transpose
    y = pt.tensor(y[..., None], dtype=pt.float)
    # y = pt.tensor(y, dtype=pt.float)
    # print(x)
    # print(y)

    #do the dataset slice
    i_train = np.s_[:train]
    i_test = np.s_[train:train+test]
    i_sample = np.s_[train+test:]
    d_train = Dataset(x[i_train], y[i_train])
    d_test = Dataset(x[i_test], y[i_test])
    d_sample = Dataset(x[i_sample], y[i_sample])
    return d_train, d_test, d_sample