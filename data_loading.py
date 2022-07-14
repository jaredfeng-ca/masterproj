import sklearn as skl
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_skl_dset(dset=None, task=None, test_split=0.5, sample_sz=None):
    assert dset is not None, "provide name of dataset!"
    assert task is not None, "provide type of task!"

    if dset == 'iris':  # n == 150, d == 4
        source = skl.datasets.load_iris()
    elif dset == 'wine':  # n == 178, d == 13
        source = skl.datasets.load_wine()

    x_train, x_test, y_train, y_test = train_test_split(source['data'],
                                                        source['target'],
                                                        test_size=test_split)

    x_train = torch.FloatTensor(x_train).t().to(device)
    y_train = torch.LongTensor(y_train).to(device)

    if task == 'class':
        y_train_oh = F.one_hot(y_train).t()
    elif task == 'reg':
        y_train_oh = None
    else:
        raise NotImplementedError('task type not implemented yet!')

    x_test = torch.FloatTensor(x_test).t().to(device)
    y_test = torch.LongTensor(y_test).to(device)

    return x_train, y_train, y_train_oh, x_test, y_test


def stratified_sampling(sample_sz=0.1, x, y):
    assert isinstance(x, np.array()), "Convert x to np array first!"
    assert isinstance(y, np.array()), "Convert y to np array first!"
    _, x, _, y = train_test_split(x, y, test_size=sample_sz, stratify=y)
    return x, y


def load_fmnist(root="./fmnist", sample_sz=0.1):
    train_data = FashionMNIST(
        root="./fmnist",
        train=True,
        download=True
    )

    test_data = FashionMNIST(
        root="./fmnist",
        train=False,
        download=True
    )

    x_train, y_train = train_data._load_data()
    x_train = x_train.view(-1, 28 * 28)

    x_test, y_test = test_data._load_data()
    x_test = x_test.view(-1, 28 * 28)

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    _, x_train, _, y_train = train_test_split(x_train, y_train,
                                              test_size=sample_sz, stratify=y_train)

    x_train = torch.Tensor(x_train).to(device)
    x_train = x_train.transpose(0, 1)
    y_train = torch.Tensor(y_train).to(device)
    y_train_oh = F.one_hot(y_train.long()).t()
    x_test = x_test.float().t().to(device)

    return x_train, y_train, y_train_oh, x_test, y_test


if __name__ == '__main__':
    print('loading some data')
    _ = load_skl_dset('wine', 'class', 0.5)