#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import random
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import print_summary
from keras.callbacks import EarlyStopping
import json
from notipy_me import Notipy


# In[2]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


# In[3]:


def set_seed(seed:int):
    """Set the random state of the various random extractions.
        seed:int, the seed to set the random state to.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


# In[4]:


def is_gpu_available():
    return bool(K.tensorflow_backend._get_available_gpus())


# In[5]:


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# In[6]:


if isnotebook():
    from tqdm import tqdm_notebook as tqdm
    from keras_tqdm import TQDMNotebookCallback as ktqdm
else: 
    from tqdm import tqdm
    from keras_tqdm import TQDMCallback as ktqdm


# In[7]:


def load_dataset(x:str, y:str)->Tuple[np.ndarray, np.ndarray]:
    return pd.read_csv(x, index_col=0).values, pd.read_csv(y, index_col=0).values


# In[8]:


def scale(train:np.ndarray, test:np.ndarray):
    """Return scaler, scaled training and test vectors based on given training vector."""
    scaler = MinMaxScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)


# In[9]:


def split_dataset(dataset:Tuple[np.ndarray, np.ndarray], seed:int, test_size:float=0.3)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the given dataset split among training a test set for the given random seed.
        dataset:Tuple[np.ndarray, np.ndarray], the dataset to split.
        seed:int, the seed to use for the random split.
    """
    return train_test_split(*dataset, test_size=test_size, random_state=seed)


# In[10]:


def scale_split_dataset(dataset, seed:int, test_size:float=0.3):
    """Return split and scaled dataset."""
    x_train, x_test, y_train, y_test = split_dataset(dataset, seed, test_size)
    return (*scale(x_train, x_test), y_train, y_test)


# In[11]:


def auprc(y_true, y_pred)->float:
    score = tf.metrics.auc(y_true, y_pred, curve="PR", summation_method="careful_interpolation")[1]
    K.get_session().run(tf.local_variables_initializer())
    return score


# In[12]:


def auroc(y_true, y_pred)->float:
    score = tf.metrics.auc(y_true, y_pred, curve="ROC", summation_method="careful_interpolation")[1]
    K.get_session().run(tf.local_variables_initializer())
    return score


# In[13]:


def mlp(input_size:int):
    """Return a multi-layer perceptron."""
    set_seed(42)
    model = Sequential([
        InputLayer(input_shape=(input_size,)),
        *[Dense(input_size, activation="relu") for i in range(5)],
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss='binary_crossentropy',
        metrics=[auprc, auroc, "accuracy"]
    )
    return model


# In[14]:


def fit(model:Sequential, dataset, holdout, epochs:int, batch_size:int):
    """Train the given model on given train data for the given epochs number.
        model:Sequential, the model to be trained.
        x_train:np.ndarray, the input for training the model.
        x_test:np.ndarray, the input for testing the model.
        y_train:np.ndarray, the output labels for training the model.
        y_test:np.ndarray, the output labels for testing the model.
        epochs:int, number of epochs for which to train the model.
        initial_epoch:int, starting epoch.
        batch_size:int, number of datapoints per training batch.
    """
    x_train, x_test, y_train, y_test = scale_split_dataset(dataset, holdout)
    return model.fit(
        x_train,
        y_train,
        shuffle=True,
        verbose=0,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[
            ktqdm(leave_inner=False, leave_outer=False),
            EarlyStopping(
                monitor="auprc",
                min_delta=0.005,
                patience=5,
                mode="max",
                restore_best_weights=True
            )
        ],
        batch_size=batch_size
    )


# In[15]:


def store_history(batch_size:int, holdout:int, auprc:float, path:str="history.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            auprcs = json.load(f)
    else:
        auprcs = {}
    if batch_size not in auprcs:
        auprcs[batch_size] = {}
    if holdout not in auprcs[batch_size]:
        auprcs[batch_size][holdout] = auprc
    with open(path, "w") as f:
        json.dump(auprcs, f)


# In[16]:


def is_history_cached(batch_size:int, holdout:int, path:str="history.json"):
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        auprcs = json.load(f)
        return batch_size in auprcs and holdout in auprcs[batch_size]


# In[17]:


def train_holdouts(batch_size:int, holdouts:int, dataset, epochs:int):
    [
        store_history(
            batch_size,
            holdout,
            fit(
                mlp(26),
                dataset,
                holdout,
                epochs, 
                batch_size
            ).history)
        for holdout in tqdm(range(holdouts), desc="Holdouts for batch_size {batch_size}".format(batch_size=batch_size), leave=False)
        if not is_history_cached(batch_size, holdout)
    ]


# In[18]:


@Notipy("./mail_configuration.json", "Batchsize experiment on Souris has completed!")
def train_batch_sizes(batch_sizes:List[int], datapoints:str, labels:str, holdouts:int, epochs:int):
    dataset = load_dataset(datapoints, labels)
    [
        train_holdouts(batch_size, holdouts, dataset, epochs) 
        for batch_size in tqdm(batch_sizes, desc="Batch sizes")
    ]


# In[21]:


def get_batch_sizes(n:int, offset:int=5):
    return [
        i**2 + int(1.175**i) for i in range(offset, n+offset)
    ]


# In[26]:


holdouts = 50
epochs = 1000
batch_sizes = get_batch_sizes(55)
print(batch_sizes)
if is_gpu_available():
    print("Working with GPU!")
train_batch_sizes(batch_sizes, "folds/x_4.csv", "folds/y_4.csv", holdouts, epochs)


# In[ ]:




