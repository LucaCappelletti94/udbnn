from .mkdir import mkdir

@mkdir
def get_history_path(path:str, batch_size:int, holdout:int):
    return "{path}/run/{batch_size}/{holdout}".format(
        path=path,
        batch_size=batch_size,
        holdout=holdout
    )