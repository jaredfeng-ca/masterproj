import warnings
from sklearn import linear_model, metrics, neighbors
from data_loading import *
from kernels import *
from dim_reductions import *


def fit_and_eval(z_train, y_train, z_test, y_test, met=None, head_type=None,
                 verbose=0):
    """
    :param z_train: optionally transformed data, (dimensions, sample size)
    :param y_train: labels, (sample size,)
    :param z_test:  optionally transformed test data, (dimensions, sample size)
    :param y_test:  test labels, (sample size,)
    :return:
    """

    assert head_type is not None, 'specify model type!'
    assert met is not None, 'specify metric type!'

    if isinstance(z_train, torch.Tensor):
        z_train = z_train.t().cpu().numpy()
    else:
        z_train = z_train.t()
    if isinstance(z_test, torch.Tensor):
        z_test = z_test.t().cpu().numpy()
    else:
        z_test = z_test.t()

    if head_type == 'logreg':
        head = linear_model.LogisticRegression()
    elif head_type == 'linreg':
        head = linear_model.LinearRegression()
    elif head_type == 'knncls':
        head = neighbors.KNeighborsClassifier()
    elif head_type == 'knnreg':
        head = neighbors.KNeighborsRegressor()
    else:
        raise NotImplementedError('model type not implemented yet!')

    if verbose:
        head.fit(z_train, y_train.cpu().numpy())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            head.fit(z_train, y_train.cpu().numpy())
    pred = head.predict(z_test)

    if met == 'accuracy':
        metric = metrics.accuracy_score
    elif met == 'mse':
        metric = metrics.mean_squared_error
    else:
        raise NotImplementedError('metric type not implemented yet!')

    return metric(pred, y_test.cpu().numpy())


if __name__ == '__main__':
    x_train, y_train, y_train_oh, x_test, y_test = load_skl_dset('wine', 'class', 0.5)
    result = fit_and_eval(x_train, y_train, x_test, y_test, met='accuracy', head_type='logreg')
    print(result)

    B, K, K_test, n = get_kerns(x_train, x_test, y_train, y_train_oh, 'class')
    z_train, z_test = kspca(K, K_test, B)
    result = fit_and_eval(z_train, y_train, z_test, y_test, met='accuracy', head_type='logreg')
    print(result)
