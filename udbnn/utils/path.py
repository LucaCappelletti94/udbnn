from .mkdir import mkdir
from typing import Dict

@mkdir
def get_history_path(path:str, batch_size:int, holdout:Dict):
    return "{path}/run/{batch_size}/{holdout_type}/{holdout}".format(
        path=path,
        batch_size=batch_size,
        holdout_type=holdout["type"],
        holdout="+".join([str(c) for c in holdout["chromosomes"]]) if holdout["type"] == "chromosomal" else holdout["seed"]
    )