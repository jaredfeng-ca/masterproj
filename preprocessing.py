import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def std_scale(x_train, x_test):
    scale = StandardScaler(with_mean=False)
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    return x_train, x_test