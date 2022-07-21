import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


def stratified_sampling(x, y, sample_sz=0.1):
    assert isinstance(x, np.ndarray), "Convert x to np array first!"
    assert isinstance(y, np.ndarray), "Convert y to np array first!"
    _, x, _, y = train_test_split(x, y, test_size=sample_sz, stratify=y)
    return x, y


def uniform_sampling(x, y, sample_sz=0.1):
    assert isinstance(x, np.ndarray), "Convert x to np array first!"
    assert isinstance(y, np.ndarray), "Convert y to np array first!"
    sample_idx = np.random.choice(len(y), int(np.floor(len(y) * 0.1)))
    x = x[sample_idx]
    y = y[sample_idx]
    return x, y