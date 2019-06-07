import numpy as np
from extra_keras_utils import set_seed

def get_batch_sizes(resolution:int, minimum:int, size:int, seed:int, base:float=1.1, delta:int=10):
    set_seed(seed)
    print(resolution, minimum, size, seed, base, delta)
    batch_sizes = base**np.arange(delta, delta+resolution)
    np.random.shuffle(batch_sizes)
    batch_sizes = minimum+np.ceil(batch_sizes/np.max(batch_sizes)*(size-minimum)).astype(int)
    batch_sizes = np.array(list(set(batch_sizes[batch_sizes<100])))
    print(batch_sizes)
    return batch_sizes