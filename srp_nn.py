import torch
from torch import nn
from data_loading import *
from kernels import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: assuming we throw all data into one batch


class SRPLayer(nn.Module):
    def __init__(self, out_dim=None, kern_approx=None, task=None):
        super(SRPLayer, self).__init__()
        self.kern_approx = kern_approx if kern_approx is not None else "rff"
        assert out_dim > 0, "Please provide a positive integer for output dimension!"
        self.out_dim = out_dim
        self.task = task
        self.Psi = None

    def __repr__(self):
        return f"SRPLayer(out_dim={self.out_dim}, kern_approx={self.kern_approx}, task={self.task})"

    def forward(self, x, x_test=None, y_train=None, y_train_oh=None):
        with torch.no_grad():
            # shape of x: (# of dims, # of examples)
            n = x.size(1)
            H = torch.eye(n) - 1/n * torch.ones(n, n)
            H = H.to(device)
            if self.training:
                if self.kern_approx == "rff":
                    self.Psi = rnd_fourier_feat(y_train, y_train_oh, self.out_dim, self.task)
                else:
                    raise NotImplementedError("Kernel approximation method not implemented yet!")
                out = self.Psi @ H @ x.t() @ x
            else:
                assert self.Psi is not None, "Please train the network first!"
                out = self.Psi @ H @ x.t() @ x, self.Psi @ H @ x.t() @ x_test
        return out


class SRPNet(nn.Module):
    def __init__(self, layer_dims=None, kern_approx=None, task=None):
        super(SRPNet, self).__init__()
        assert layer_dims is not None, "Please provide dimension (in integers) of each layer in a list, " \
                                       "e.g. [16, 8, 4] for a 3-layer SRPNet."
        self.num_layers = len(layer_dims)
        self.srp_layers = nn.ModuleList([SRPLayer(out_dim=layer_dim, kern_approx=kern_approx, task=task)
                                        for layer_dim in layer_dims])
        self.training = True
        for srp_layer in self.srp_layers:
            srp_layer.requires_grad_(False)

    def forward(self, x_train, x_test=None, y_train=None, y_train_oh=None):
        if self.training:
            assert y_train is not None or y_train_oh is not None, "Please provide labels!"
            z_train = x_train
            for srp_layer in self.srp_layers:
                z_train = srp_layer(z_train, y_train=y_train, y_train_oh=y_train_oh)
            return z_train
        else:
            assert x_test is not None, "Please provide test data!"
            z_train, z_test = x_train, x_test
            for srp_layer in self.srp_layers:
                z_train, z_test = srp_layer(z_train, x_test=z_test)
            return z_test


def srpnn_dim_reduct(x_train, x_test, y_train=None, y_train_oh=None, kern_approx=None,
                     layer_dims=None, task=None):
    net = SRPNet(layer_dims=layer_dims, kern_approx=kern_approx, task=task).to(device)
    net.train()
    z_train = net(x_train, y_train=y_train, y_train_oh=y_train_oh)
    net.eval()
    z_test = net(x_train, x_test=x_test)
    return z_train, z_test


if __name__ == "__main__":
    task = 'class'
    dset = 'wine'
    x_train, y_train, y_train_oh, x_test, y_test = load_skl_dset(dset, task, 0.3)
    net = SRPNet(layer_dims=[16, 6, 1], task=task).to(device)
    net.train()
    z_train = net(x_train, y_train_oh=y_train_oh)
    net.eval()
    z_test = net(x_train, x_test=x_test)
    print(net)
    print(z_train.size())
    print(z_test.size())

