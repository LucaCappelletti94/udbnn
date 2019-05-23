from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from .utils import set_seed, auprc, auroc

def model(input_size:int):
    """Return a multi-layer perceptron."""
    set_seed(42)
    model = Sequential([
        InputLayer(input_shape=(input_size,)),
        *[Dense(20, activation="relu") for i in range(2)],
        Dropout(0.5),
        *[Dense(10, activation="relu") for i in range(2)],
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss='binary_crossentropy',
        metrics=[auprc, auroc, "accuracy"]
    )
    return model