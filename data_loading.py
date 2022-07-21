# TODO: create a function to preprocess csv's one-off, change csv paths under code to store the processed csvs
# TODO: might need to separate the preprocessing function(s) to a new .py
# TODO: or just simply find cleaned datasets

import sklearn as skl
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import gc

import torch
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

from preprocessing import *
from sampling import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_skl_dset(dset=None, task=None, test_split=0.5, sample_sz=None):
    assert dset is not None, "provide name of dataset!"
    assert task is not None, "provide type of task!"

    if dset == 'iris':  # n == 150, d == 4, 3 class
        source = skl.datasets.load_iris()
    elif dset == 'wine':  # n == 178, d == 13, 3 class
        source = skl.datasets.load_wine()
    elif dset == 'digits':  # n == 1797, d == 64, 10 class
        source = skl.datasets.load_digits()
    elif dset == 'diabetes':  # n == 442, d == 10, reg
        source = skl.datasets.load_diabetes()
    elif dset == 'boston':  # n == 506, d == 13, reg
        source = skl.datasets.load_boston()
    elif dset == 'bcancer':  # n == 569, d == 30, 2 class
        source = skl.datasets.load_breast_cancer()
    else:
        raise NotImplementedError('dataset not implemented yet!')

    x_train, x_test, y_train, y_test = train_test_split(source['data'],
                                                        source['target'],
                                                        test_size=test_split)
    x_train, x_test = std_scale(x_train, x_test)
    x_train = torch.FloatTensor(x_train).t().to(device)
    x_test = torch.FloatTensor(x_test).t().to(device)

    if task == 'class':
        y_train = torch.LongTensor(y_train).to(device)
        y_train_oh = F.one_hot(y_train).t()
        y_test = torch.LongTensor(y_test).to(device)
    elif task == 'reg':
        y_train_oh = None
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    else:
        raise NotImplementedError('task type not implemented yet!')

    return x_train, y_train, y_train_oh, x_test, y_test


def load_csv_dset(path=None, dset=None, task=None, test_split=0.5, sample_sz=None):
    assert dset is not None, "provide name of dataset!"
    assert task is not None, "provide type of task!"

    df = pd.read_csv(path, encoding_errors='ignore')

    if dset == 'heart':
        y = df['target'].to_numpy()
        x = df.drop(columns='target').to_numpy()
    if dset == 'careval':
        y = df.iloc[:, -1].to_numpy()
        x = df.iloc[:, :-1].to_numpy()
    if dset == 'gaspx':
        y = df['Price Per Gallon (USD)'].to_numpy()
        df_x = df.drop(columns=['Price Per Gallon (USD)',
                                'Country',
                                'S#',
                                'Price Per Liter (USD)',
                                'Price Per Liter (PKR)'])
        df_x['World Share'] = df_x['World Share'].apply(lambda h: float(h.replace("%", "")))/100
        df_x.iloc[:, [0, 3, 4]] = df_x.iloc[:, [0, 3, 4]].applymap(lambda h: float(h.replace(",", "")))
        x = df_x.to_numpy()

    del df
    gc.collect()

    # sample down
    if sample_sz is not None:
        if task == "class":
            x, y = stratified_sampling(x, y, sample_sz)
        elif task == "reg":
            x, y = uniform_sampling(x, y, sample_sz)

    # feature engineering (encoding cat features + scaling num features)
    oh_enc = OneHotEncoder(sparse=False)
    x = oh_enc.fit_transform(x)

    # train test split on subset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)

    x_train = torch.FloatTensor(x_train).t().to(device)
    x_test = torch.FloatTensor(x_test).t().to(device)

    if task == 'class':
        y_train = torch.LongTensor(y_train).to(device)
        y_train_oh = F.one_hot(y_train).t()
        y_test = torch.LongTensor(y_test).to(device)
    elif task == 'reg':
        y_train_oh = None
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    else:
        raise NotImplementedError('task type not implemented yet!')

    return x_train, y_train, y_train_oh, x_test, y_test


def load_fmnist(root="../data/fmnist", sample_sz=0.1):
    train_data = FashionMNIST(
        root=root,
        train=True,
        download=True
    )

    test_data = FashionMNIST(
        root=root,
        train=False,
        download=True
    )

    x_train, y_train = train_data._load_data()
    x_train = x_train.view(-1, 28 * 28)

    x_test, y_test = test_data._load_data()
    x_test = x_test.view(-1, 28 * 28)

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_train, y_train = stratified_sampling(x_train, y_train, 0.1)

    x_train, x_test = std_scale(x_train, x_test)

    x_train = torch.Tensor(x_train).to(device)
    x_train = x_train.transpose(0, 1)
    y_train = torch.Tensor(y_train).to(device)
    y_train_oh = F.one_hot(y_train.long()).t()
    x_test = x_test.float().t().to(device)

    return x_train, y_train, y_train_oh, x_test, y_test


if __name__ == '__main__':
    print('loading some data')
    # _ = load_skl_dset('bcancer', 'class', 0.5)
    # _ = load_fmnist(root='../data/fmnist', sample_sz=0.1)
    # load_csv_dset('../data/gaspx/Petrol Dataset June 23 2022 -- Version 2.csv', 'gaspx',
    #               'reg', 0.5, 0.5)