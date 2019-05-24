from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from .utils import set_seed, auprc, auroc
from keras.initializers import RandomNormal

def model(input_size:int):
    """Return a multi-layer perceptron."""
    set_seed(42)
    model = Sequential([
        InputLayer(input_shape=(input_size,)),
        *[Dense(20, activation="selu", kernel_initializer=RandomNormal(mean=0, stddev=0.05), bias_initializer='zeros') for i in range(10)],
        Dropout(0.5),
        #*[Dense(10, activation="relu") for i in range(2)],
        #Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss='binary_crossentropy',
        metrics=[auprc, auroc, "accuracy"]
    )
    return model