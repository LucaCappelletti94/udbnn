from os.path import exists
from typing import Dict
from .path import get_history_path
from .get_batch_sizes import get_batch_sizes
from .dataset import load_dataset

def is_holdout_cached(path:str, batch_size:int, holdout:Dict)->bool:
    return exists("{path}/history.json".format(path=get_history_path(path, batch_size, holdout)))

def is_batch_size_cached(path:str, batch_size:int, settings:Dict)->bool:
    return all([
        is_holdout_cached(path, batch_size, holdout) for holdout in settings["holdouts"]
    ])

def is_dataset_cached(path:str, settings:Dict)->bool:
    x, y = load_dataset(path, settings["max_correlation"])
    return all([
        is_batch_size_cached(path, batch_size, settings) for batch_size in get_batch_sizes(
            resolution=settings["batch_sizes"]["resolution"],
            size=x.shape[0],
            seed=settings["batch_sizes"]["seed"]
        )
    ])