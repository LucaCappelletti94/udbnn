import numpy as np
from .set_seed import set_seed

def get_batch_sizes(resolution:int, size:int, seed:int):
    set_seed(seed)
    batch_sizes = np.sort(np.random.exponential(scale=1.0, size=resolution))
    return np.ceil(batch_sizes/np.max(batch_sizes)*size).astype(int)
