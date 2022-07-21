import torch
import warnings
import numpy as np
from data_loading import *
from kernels import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pca(x_train, x_test, output_dim=2):

    with torch.no_grad():
        n = x_train.size(1)
        H = torch.eye(n) - 1/n * torch.ones(n, n)
        H = H.to(device)

        Q = x_train @ H @ H @ x_train.t()
        L, V = torch.linalg.eig(Q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = L.float()
            V = V.float()

        U = V[:, :output_dim]

        # encode training data
        z_train = U.t() @ x_train

        # encode test data
        z_test = U.t() @ x_test

    return z_train, z_test


def spca(x_train, x_test, B, output_dim=2):

    with torch.no_grad():
        n = x_train.size(1)
        H = torch.eye(n) - 1 / n * torch.ones(n, n)
        H = H.to(device)

        # SPCA, Q = XHBHX^t
        Q = x_train @ H @ B @ H @ x_train.t()
        L, V = torch.linalg.eig(Q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = L.float()
            V = V.float()

        U = V[:, :output_dim]

        # encode training data
        z_train = U.t() @ x_train

        # encode test data
        z_test = U.t() @ x_test

    return z_train, z_test


def kspca(K, K_test, B, output_dim=2):

    with torch.no_grad():
        n = K.size(0)
        H = torch.eye(n) - 1 / n * torch.ones(n, n)
        H = H.to(device)

        Q = H @ B @ H @ K
        L, V = torch.linalg.eig(Q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = L.float()
            V = V.float()

        # compute basis
        U = V[:, :output_dim]

        # encode training data
        z_train = U.t() @ K

        # encode test data
        z_test = U.t() @ K_test

    return z_train, z_test


def srp(x_train, x_test, y_train=None, y_train_oh=None, kern_approx=None, output_dim=2, task=None):
    n = x_train.size(1)
    with torch.no_grad():
        H = torch.eye(n) - 1 / n * torch.ones(n, n)
        H = H.to(device)

        if kern_approx == 'rff' or kern_approx is None:
            Psi = rnd_fourier_feat(y_train, y_train_oh, output_dim, task)
        else:
            raise NotImplementedError("Kernel approximation method not implemented yet!")

        # encode training data. output size: (2k, n)
        z_train = Psi @ H @ x_train.t() @ x_train
        # encode test data. output size: (2k, n_test)
        z_test = Psi @ H @ x_train.t() @ x_test
    return z_train, z_test


def ksrp(K, K_test, y_train=None, y_train_oh=None, kern_approx=None, output_dim=2, task=None):
    n = K.size(0)
    with torch.no_grad():
        H = torch.eye(n) - 1 / n * torch.ones(n, n)
        H = H.to(device)

        if kern_approx == 'rff' or kern_approx is None:
            Psi = rnd_fourier_feat(y_train, y_train_oh, output_dim, task)
        else:
            raise NotImplementedError("Kernel approximation not implemented yet!")

        # encode training data. output size: (2k, n)
        z_train = Psi @ H @ K
        # encode test data. output size: (2k, n_test)
        z_test = Psi @ H @ K_test
    return z_train, z_test

# def ksrp_approx():


if __name__ == '__main__':
    task = 'class'
    dset = 'wine'
    x_train, y_train, y_train_oh, x_test, y_test = load_skl_dset(dset, task)
    B, K, K_test, n = get_kerns(x_train, x_test, y_train, y_train_oh, task)
    #z_train, z_test = pca(x_train, x_test)
    #z_train, z_test = spca(x_train, x_test, B)
    #z_train, z_test = kspca(K, K_test, B)
    #z_train, z_test = srp(x_train, x_test, y_train_oh)
    z_train, z_test = ksrp(K, K_test, y_train_oh)

