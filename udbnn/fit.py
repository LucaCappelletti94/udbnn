from typing import Tuple, Dict
from .utils import scale_split_dataset, ktqdm
from .model import model
import pandas as pd
from keras.callbacks import EarlyStopping

def fit(dataset:Tuple[pd.DataFrame, pd.DataFrame], batch_size:int, holdout:int, settings:Dict):
    """Train the given model on given train data for the given epochs number.
        dataset:Tuple[pd.DataFrame, pd.DataFrame], the dataset to split into train and test set.
        batch_size:int, size for the batch size of this run.
        holdout:int, the seed for the split        
        settings:Dict, the training settings.
    """
    x_train, x_test, y_train, y_test = scale_split_dataset(dataset, holdout)
    return model(x_train.shape[1]).fit(
        x_train,
        y_train,
        verbose=0,
        validation_data=(x_test, y_test),
        **settings["training"]["fit"],
        batch_size=batch_size,
        callbacks=[
            ktqdm(),
            EarlyStopping(**settings["training"]["early_stopping"])
        ]
    )