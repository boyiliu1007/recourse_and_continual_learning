import sklearn as skl
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch as pt
from torch.utils import data
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

def load_credit_default_data():
    pt.manual_seed(0)
    np.random.seed(0)
    url = 'https://raw.githubusercontent.com/ustunb/actionable-recourse/master/examples/paper/data/credit_processed.csv'
    df = pd.read_csv(url)
    df = df.sample(frac=1).reset_index(drop=True)

    df = df.drop(['Married', 'Single', 'Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60'], axis = 1)

    scaler = StandardScaler()
    df.loc[:, df.columns != "NoDefaultNextMonth"] = scaler.fit_transform(df.drop("NoDefaultNextMonth", axis=1))

    fraud_df = df.loc[df["NoDefaultNextMonth"] == 0]
    non_fraud_df = df.loc[df["NoDefaultNextMonth"] == 1][:6636]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    df = normal_distributed_df.sample(frac=1).reset_index(drop=True)

    Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
    
    return X, Y

def load_german_data():
    df = pd.read_csv('Dataset\german.csv')
    df = df.drop(['Status', 'History', 'Present residence', 'Age', 'Number people'], axis = 1)
    X = df.drop('label', axis=1).values
    Y = df['label'].values
    print(X.shape)
    print(Y.shape)
    return X, Y

def load_sba_data():
    df = pd.read_csv('Dataset\sba.csv')
    df = df[['Selected','Term', 'NoEmp','CreateJob', 'RetainedJob']]
    X = df.drop('Selected', axis=1).values
    Y = df['Selected'].values
    
    print(X.shape)
    print(Y.shape)
    return X, Y

DATASETS = ['synthetic', 'credit', 'german', 'sba']
def make_dataset(train: int, test: int, sample: int, positive_ratio: float = 0.5, dataset: str = 'synthetic'):
    n_samples = train + test + sample

    if dataset == 'synthetic':
        #generate dataset with n_sameples data and 20 features
        x, y = make_classification(n_samples, weights=[1 - positive_ratio, positive_ratio], random_state=42)
        x = pt.tensor(x, dtype=pt.float)
        y = pt.tensor(y[..., None], dtype=pt.float).squeeze()
        
        
    
    if dataset == 'credit':
        X, Y = load_credit_default_data()
        X, Y = X[:n_samples], Y[:n_samples]
        x = pt.tensor(X, dtype=pt.float).clone().detach()
        y = pt.tensor(Y, dtype=pt.float).clone().detach()
        

    if dataset == 'german':
        X, Y = load_german_data()
        X, Y = X[:n_samples], Y[:n_samples]
        x = pt.tensor(X, dtype=pt.float).clone().detach()
        y = pt.tensor(Y, dtype=pt.float).clone().detach()
        

    if dataset == 'sba':
        X, Y = load_sba_data()
        X, Y = X[:n_samples], Y[:n_samples]
        x = pt.tensor(X, dtype=pt.float).clone().detach()
        y = pt.tensor(Y, dtype=pt.float).clone().detach()
        
        

    #do the dataset slice
    i_train = np.s_[:train]
    i_test = np.s_[train:train+test]
    i_sample = np.s_[train+test:]
    d_train = Dataset(x[i_train], y[i_train])
    d_test = Dataset(x[i_test], y[i_test])
    d_sample = Dataset(x[i_sample], y[i_sample])
    return d_train, d_test, d_sample


