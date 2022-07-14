import torch
import gpytorch
from data_loading import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_kerns(x_train, x_test, labels=None, oh_labels=None, task=None):
    """
    x_train: d x n float tensor
    y_train: 1 x n tensor
    y_train_oh:  p x n long tensor, where p is the number of classes
    """

    assert task is not None, "provide type of task!"

    n = x_train.size(1)

    get_lin_kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    get_gau_kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    if task == 'class':
        assert oh_labels is not None, "provide one hot labels for kernel calculation!"
        B = get_lin_kern(oh_labels.float().t()).to(device).detach()
    elif task == 'reg':
        assert labels is not None, "provide labels for kernel calculation!"
        B = get_gau_kern(labels.t()).to(device).detach()
    else:
        raise NotImplementedError('task type not implemented yet!')

    K = get_gau_kern(x_train.t()).to(device).detach()
    K_test = get_gau_kern(x_train.t(), x_test.t()).to(device).detach()

    return B, K, K_test, n


if __name__ == '__main__':
    print('loading some data ... ')

    x_train, y_train, y_train_oh, x_test, y_test = load_skl_dset('wine', 'class')

    print('data loaded. Calculating kernels ... ')

    B, K, K_test, n = get_kerns(x_train, x_test, y_train, y_train_oh, 'class')

    print('kernels calculated.')
