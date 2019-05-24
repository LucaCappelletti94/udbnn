import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .ungzip import ungzip
from os.path import exists
import numpy as np

def load_dataset(path:str, max_correlation:float)->Tuple[pd.DataFrame, pd.DataFrame]:
    if not exists("{path}/x.csv".format(path=path)):
        ungzip("{path}/x.csv.gz".format(path=path))
    if not exists("{path}/y.csv".format(path=path)):
        ungzip("{path}/y.csv.gz".format(path=path))
    x = pd.read_csv("{path}/x.csv".format(path=path), index_col=0).astype(float)
    x = x.drop(columns=x.columns[np.any(np.triu(x.corr()>max_correlation, k=1), axis=1)])
    return (x, pd.read_csv("{path}/y.csv".format(path=path), index_col=0))

def scale(train:pd.DataFrame, test:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
    """Return scaler, scaled training and test vectors based on given training vector."""
    scaler = MinMaxScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)

def split_dataset(dataset:Tuple[pd.DataFrame, pd.DataFrame], holdout:Dict, test_size:float=0.3)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the given dataset split among training a test set for the given random holdout.
        dataset:Tuple[np.ndarray, np.ndarray], the dataset to split.
        holdout:Dict, the holdout to use for the random split.
    """
    if holdout["type"] == "random":
        return train_test_split(*dataset, test_size=test_size, random_state=holdout["seed"])
    else:
        x, y = dataset
        chromosomes = ["chr{chromosome}".format(chromosome=chromosome) for chromosome in holdout["chromosomes"]]
        mask = np.array([i.split(".")[0] in chromosomes for i in x.index])
        return x[~mask], x[mask], y[~mask], y[mask]

def scale_split_dataset(dataset, holdout:Dict, test_size:float=0.3):
    """Return split and scaled dataset."""
    x_train, x_test, y_train, y_test = split_dataset(dataset, holdout, test_size)
    return (*scale(x_train, x_test), y_train, y_test)