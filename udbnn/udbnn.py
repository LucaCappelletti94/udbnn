from typing import Dict
import pandas as pd
from typing import Tuple
from .fit import fit
from .utils import story_history, get_history_path, is_dataset_cached, is_batch_size_cached, is_holdout_cached, load_dataset, load_settings, tqdm, is_gpu_available, is_tmux, get_batch_sizes
from notipy_me import Notipy

def train_holdout(path:str, dataset:Tuple[pd.DataFrame, pd.DataFrame], batch_size:int, settings:Dict):
    holdouts = [
        holdout for holdout in settings["holdouts"]
        if not is_holdout_cached(path, batch_size, holdout)
    ]
    for holdout in tqdm(holdouts, desc="Holdouts", leave=False):
        story_history(
            get_history_path(path, batch_size, holdout),
            fit(dataset, batch_size, holdout, settings)
        )

def train_batch_sizes(dataset_path:str, settings:Dict):
    dataset = load_dataset(dataset_path, settings["max_correlation"])
    batch_sizes = [
        v for v in get_batch_sizes(
            resolution=settings["batch_sizes"]["resolution"],
            size=dataset[0].shape[0],
            seed=settings["batch_sizes"]["seed"]
        ) if not is_batch_size_cached(dataset_path, v, settings)
    ]
    for batch_size in tqdm(batch_sizes, desc="Batch sizes", leave=False):
        train_holdout(dataset_path, dataset, batch_size, settings)

def train_datasets(target:str):
    settings = load_settings(target)
    datasets = [
        "{target}/{path}".format(target=target, path=dataset["path"])
        for dataset in settings["datasets"]
        if dataset["enabled"] and not is_dataset_cached("{target}/{path}".format(target=target, path=dataset["path"]), settings)
    ]
    for path in tqdm(datasets, desc="Datasets"): 
        train_batch_sizes(path, settings)

#@Notipy("./mail_configuration.json", "Batchsize experiment has completed!")
def run(target:str):
    if not is_gpu_available():
        print("No GPU was detected!")
    if not is_tmux():
        print("Not running within TMUX!")
    train_datasets(target)