from keras import backend as K
import tensorflow as tf

def auprc(y_true, y_pred)->float:
    score = tf.metrics.auc(y_true, y_pred, curve="PR", summation_method="careful_interpolation")[1]
    K.get_session().run(tf.local_variables_initializer())
    return score

def auroc(y_true, y_pred)->float:
    score = tf.metrics.auc(y_true, y_pred, curve="ROC", summation_method="careful_interpolation")[1]
    K.get_session().run(tf.local_variables_initializer())
    return score