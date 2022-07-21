import torch
import gpytorch
from data_loading import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_kerns(x_train, x_test, labels=None, oh_labels=None, task=None):
    """
    x_train: d x n float tensor
    x_test: d x n_test float tensor
    labels: 1 x n tensor
    oh_labels:  p x n long tensor, where p is the number of classes
    """

    assert task is not None, "provide type of task!"

    n = x_train.size(1)

    # get_lin_kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    get_gau_kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    if task == 'class':
        assert oh_labels is not None, "provide one hot labels for kernel calculation!"
        B = torch.eq(oh_labels.t().unsqueeze(1), oh_labels.t().unsqueeze(0)).prod(dim=-1)
        B = B.float()
    elif task == 'reg':
        assert labels is not None, "provide labels for kernel calculation!"
        B = get_gau_kern(labels.t()).to(device).detach()
    else:
        raise NotImplementedError('task type not implemented yet!')

    K = get_gau_kern(x_train.t()).to(device).detach()
    K_test = get_gau_kern(x_train.t(), x_test.t()).to(device).detach()

    return B, K, K_test, n


def rnd_fourier_feat(y_train=None, y_train_oh=None, output_dim=2, task=None, sigma=1):
    with torch.no_grad():
        if task == 'class':
            p = y_train_oh.size(0)
        elif task == 'reg':
            p = 1
        else:
            raise Exception("Please enter task type!")

        k = output_dim
        n = y_train.size(0)

        # approximate Psi
        sigma = 1
        W = torch.randn(k, p) / sigma
        W = W.to(device)
        b = 2 * torch.pi * torch.rand(k, 1)
        b = b.to(device)
        ones = torch.ones(1, n).to(device)
        y_train = y_train.view(1, -1)

        if task == 'class':
            affine = W @ y_train_oh.float() + b @ ones
        elif task == 'reg':
            affine = W @ y_train.float() + b @ ones

        Psi = np.sqrt(2 / k) * torch.cat((torch.cos(affine), torch.sin(affine)), dim=0)
    return Psi


if __name__ == '__main__':
    print('loading some data ... ')

    x_train, y_train, y_train_oh, x_test, y_test = load_skl_dset('wine', 'class', 0.7)

    print('data loaded. Calculating kernels ... ')

    B, K, K_test, n = get_kerns(x_train, x_test, y_train, y_train_oh, 'class')
    print(K_test.size())

    print('kernels calculated.')
