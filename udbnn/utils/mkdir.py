from typing import Callable
import os

def mkdir(func:Callable):
    def wrapper(*args):
        path = func(*args)
        os.makedirs(path, exist_ok=True)
        return path
    return wrapper