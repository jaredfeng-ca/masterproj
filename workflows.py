from data_loading import *
from kernels import *
from dim_reductions import *
from modeling import *
from seed import seed_everything

import torch
import numpy as np
import gc


def single_trial_workflow(seed=None, odim=None, algo_name=None, dset=None,
                          task=None, head_type=None, met=None):
    """
    A seeded single-iteration workflow of data loading, kernel calculation,
        dimensionality reduction, model fit and evaluate
    """

    assert seed is not None, "provide seed!"
    assert odim is not None, "provide output dimension!"
    assert algo_name is not None, "provide dimension reduction algorithm!"

    # set seed
    seed_everything(seed)

    # load data
    if dset == 'fmnist':
        x_train, y_train, y_train_oh, x_test, y_test = load_fmnist(root='../data/fmnist', sample_sz=0.1)
    else:
        x_train, y_train, y_train_oh, x_test, y_test = load_skl_dset(dset, task=task)

    # kernel calculation with gpytorch
    B, K, K_test, n = get_kerns(x_train, x_test, oh_labels=y_train_oh, task=task)

    # dimensionality reduction
    if algo_name == 'pca':
        z_train, z_test = pca(x_train, x_test, odim)
    elif algo_name == 'spca':
        z_train, z_test = spca(x_train, x_test, B, odim)
    elif algo_name == 'kspca':
        z_train, z_test = kspca(K, K_test, B, odim)
    elif algo_name == 'srp':
        z_train, z_test = srp(x_train, x_test, y_train_oh, output_dim=odim)
    elif algo_name == 'ksrp':
        z_train, z_test = ksrp(K, K_test, y_train_oh, output_dim=odim)
    else:
        raise NotImplementedError('dimension reduction algorithm not implemented yet!')

    # model fit and evaluate
    return fit_and_eval(z_train, y_train, z_test, y_test, met=met,
                          head_type=head_type, verbose=0)


def multi_trial(num_trials=10, odim=2, algo_name=None, dset=None,
                task=None, head_type=None, met=None):
    num_trials = 10
    results = np.zeros(num_trials)

    for i in range(num_trials):
        res = single_trial_workflow(seed=i, odim=odim, algo_name=algo_name, dset=dset,
                                    task=task, head_type=head_type, met=met)
        results[i] = res

    return np.mean(results)


