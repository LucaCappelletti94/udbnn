import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .ungzip import ungzip
from .is_cached import is_cached

def load_dataset(path:str)->Tuple[pd.DataFrame, pd.DataFrame]:
    if not is_cached("{path}/x.csv".format(path=path)):
        ungzip("{path}/x.csv.gz".format(path=path))
    if not is_cached("{path}/y.csv".format(path=path)):
        ungzip("{path}/y.csv.gz".format(path=path))
    return (
        pd.read_csv("{path}/x.csv".format(path=path), index_col=0).astype(float),
        pd.read_csv("{path}/y.csv".format(path=path), index_col=0)
    )

def scale(train:pd.DataFrame, test:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
    """Return scaler, scaled training and test vectors based on given training vector."""
    scaler = MinMaxScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)

def split_dataset(dataset:Tuple[pd.DataFrame, pd.DataFrame], seed:int, test_size:float=0.3)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the given dataset split among training a test set for the given random seed.
        dataset:Tuple[np.ndarray, np.ndarray], the dataset to split.
        seed:int, the seed to use for the random split.
    """
    return train_test_split(*dataset, test_size=test_size, random_state=seed)

def scale_split_dataset(dataset, seed:int, test_size:float=0.3):
    """Return split and scaled dataset."""
    x_train, x_test, y_train, y_test = split_dataset(dataset, seed, test_size)
    return (*scale(x_train, x_test), y_train, y_test)