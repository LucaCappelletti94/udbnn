import pandas as pd
from keras.callbacks import History

def story_history(path:str, history:History):
    with open("{path}/history.json".format(path=path), "w") as f:
        pd.DataFrame(history.history).to_json(f)