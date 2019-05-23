from .utils import load_settings
import shutil

def clear(target:str):
    for dataset in load_settings(target)["datasets"]:
        shutil.rmtree("{target}/{path}/run".format(target=target, path=dataset["path"]))