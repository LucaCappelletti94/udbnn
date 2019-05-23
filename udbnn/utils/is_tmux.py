import os

def is_tmux():
    try:
        os.environ["TMUX"]
        return True
    except KeyError:
        return False