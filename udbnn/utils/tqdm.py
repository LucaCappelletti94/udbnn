from tqdm import tqdm_notebook, tqdm as tqdm_cli
from keras_tqdm import TQDMNotebookCallback, TQDMCallback
from .is_notebook import is_notebook

_tqdm = tqdm_notebook if is_notebook() else tqdm_cli
_ktqdm = TQDMNotebookCallback if is_notebook() else TQDMCallback

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, dynamic_ncols=True)

def ktqdm(**kwargs):
    return _ktqdm(leave_inner=False, leave_outer=False, **kwargs)