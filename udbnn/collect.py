import pandas as pd
from glob import glob
from plot_keras_history import plot_history
import os

def customize_axis(axis):
    axis.set_xscale("log")

def collect(target:str):
    histories = list(glob("{target}/**/history.json".format(target=target), recursive=True))
    dfs = []
    for history in histories:
        tail = pd.read_json(history).tail(1)
        tail.index = [int(history.split("/")[-4])]
        dfs.append(tail)
        
    concat = pd.concat(dfs).sort_index()
    concat.index.name = "Batch sizes"
    os.makedirs("history", exist_ok=True)
    plot_history(concat, customization_callback=customize_axis, path="history", single_graphs=True)
    os.makedirs("interpolated_history", exist_ok=True)
    plot_history(concat, interpolate=True, customization_callback=customize_axis, path="interpolated_history", single_graphs=True)
    