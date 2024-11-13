import sklearn as skl
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
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

    # x = pt.tensor(x, dtype=pt.float)
    # #do the transpose
    # y = pt.tensor(y[..., None], dtype=pt.float)
    
    # y = pt.tensor(y, dtype=pt.float)
    # print(x)
    # print(y)

    # #do the dataset slice
    # i_train = np.s_[:train]
    # i_test = np.s_[train:train+test]
    # i_sample = np.s_[train+test:]
    # d_train = Dataset(x[i_train], y[i_train])
    # d_test = Dataset(x[i_test], y[i_test])
    # d_sample = Dataset(x[i_sample], y[i_sample])
    # x = x.astype(np.float32)
    # y = y.astype(np.float32)
    #0.7 0.85
    
    #把資料隨機抽取並分配
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.7, random_state=42)
    X_test, X_sample, y_test, y_sample = train_test_split(X_temp, y_temp, test_size=0.9375, random_state=42)  # 0.9375 * 0.8 = 7500 / 10000

    X_train = pt.tensor(X_train, dtype=pt.float)
    X_test = pt.tensor(X_test, dtype=pt.float)
    X_sample = pt.tensor(X_sample, dtype=pt.float)
    y_train = pt.tensor(y_train, dtype=pt.float)
    y_test = pt.tensor(y_test, dtype=pt.float)
    y_sample = pt.tensor(y_sample, dtype=pt.float)

    #轉換為Dataset
    d_train = Dataset(X_train, y_train)
    d_test = Dataset(X_test, y_test)
    d_sample = Dataset(X_sample, y_sample)
    
    return d_train, d_test, d_sample