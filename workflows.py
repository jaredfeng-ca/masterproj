from data_loading import *
from kernels import *
from dim_reductions import *
from srp_nn import *
from modeling import *
from seed import seed_everything

import torch
import numpy as np
import gc
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def single_trial_workflow(seed=None, odim=None, algo_name=None, dset=None,
                          task=None, head_type=None, met=None, kern_approx=None):
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
    B, K, K_test, n = get_kerns(x_train, x_test, labels=y_train, oh_labels=y_train_oh, task=task)

    # dimensionality reduction
    if len(algo_name.split("+")) == 2:
        [algo_name, kern_approx] = algo_name.split("+")
    elif len(algo_name.split("+")) == 3:
        [algo_name, kern_approx, layer_dims] = algo_name.split("+")
        if layer_dims == "0":
            layer_dims = []
        else:
            layer_dims = [int(layer_dim) for layer_dim in layer_dims.split("_")]
        layer_dims += [odim]

    if algo_name == 'pca':
        z_train, z_test = pca(x_train, x_test, odim)
    elif algo_name == 'spca':
        z_train, z_test = spca(x_train, x_test, B, odim)
    elif algo_name == 'kspca':
        z_train, z_test = kspca(K, K_test, B, odim)
    elif algo_name == 'srp':
        z_train, z_test = srp(x_train, x_test, y_train, y_train_oh, kern_approx, odim, task)
    elif algo_name == 'ksrp':
        z_train, z_test = ksrp(K, K_test, y_train, y_train_oh, kern_approx, odim, task)
    elif algo_name == 'srpnn':
        z_train, z_test = srpnn_dim_reduct(x_train, x_test, y_train, y_train_oh, kern_approx, layer_dims, task)
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


def experiment(trials, max_out_dim, dset, task, head, metric, algos):
    out_dims = np.arange(1, max_out_dim+1)
    res_mat = np.zeros((len(algos), max_out_dim))

    for j, algo in enumerate(algos):
        algo_results = np.zeros(max_out_dim)
        for i, dim in enumerate(out_dims):
            res = multi_trial(num_trials=trials, odim=dim, algo_name=algo,
                              dset=dset, task=task, head_type=head, met=metric)
            algo_results[i] = res
        res_mat[j] = algo_results

    return res_mat.transpose(), out_dims


def experiment2(trials, out_dims, dset, task, head, metric, algos):
    res_mat = np.zeros((len(algos), len(out_dims)))

    for j, algo in enumerate(algos):
        algo_results = np.zeros(len(out_dims))
        for i, dim in enumerate(out_dims):
            res = multi_trial(num_trials=trials, odim=dim, algo_name=algo,
                              dset=dset, task=task, head_type=head, met=metric)
            algo_results[i] = res
        res_mat[j] = algo_results

    return res_mat.transpose(), out_dims


def plot_exp_result(res_mat, out_dims, dset, head, metric, algos):
    # change result array into a df for sns plot
    res_df = pd.DataFrame(res_mat, columns=algos, index=out_dims)
    res_df['dimensions'] = out_dims

    sns.set_theme()
    sns.set(font_scale=1.2)
    for algo in algos:
        plot = sns.lineplot(x='dimensions', y=algo, data=res_df, lw=2.5)
    plt.legend(labels=algos, facecolor='white')
    plot.set(ylabel=metric, title = f'{dset} dataset with model {head}, by dimensions');

