import torch
from torch import nn
import warnings
import numpy as np
from data_loading import *
from kernels import *
from dim_reductions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: assuming we throw all data into one batch


class SRPLayer(nn.Module):
    def __init__(self, kern_approx=None, out_dim=None, task=None):
        super(SRPLayer, self).__init__()
        self.kern_approx = kern_approx
        self.out_dim = out_dim
        self.task = task
        self.Psi = None

    def forward(self, x, x_test=None, y_train=None, y_train_oh=None, train=True):
        with torch.no_grad():
            # shape of x: (# of dims, # of examples)
            n = x.size(1)
            H = torch.eye(n) - 1/n * torch.ones(n, n)
            if train:
                if self.kern_approx == 'rff' or self.kern_approx is None:
                    self.Psi = rnd_fourier_feat(y_train, y_train_oh, self.out_dim, self.task)
                else:
                    raise NotImplementedError("Kernel approximation method not implemented yet!")
                out = self.Psi @ H @ x.t() @ x
            else:
                out = self.Psi @ H @ x.t() @ x_test
        return out


class SRPNet(nn.Module):
    def __init__(self, task=None):
        super(SRPNet, self).__init__()
        self.srp1 = SRPLayer(kern_approx='rff', out_dim=16, task=task)
        self.srp2 = SRPLayer(kern_approx='rff', out_dim=4, task=task)
        self.srp1.requires_grad_(False)
        self.srp2.requires_grad_(False)

    def forward(self, x, x_test=None, y_train=None, y_train_oh=None, train=True):
        if train:
            assert y_train is not None or y_train_oh is not None, "please provide labels!"
        else:
            assert x_test is not None, "please provide test data!"
        z = self.srp1(x, x_test, y_train, y_train_oh, train)
        z = self.srp2(z, x_test, y_train, y_train_oh, train)
        return z


if __name__ == "__main__":
    net = SRPNet('class')