import numpy as np
import tensorflow as tf

def calculate_threshold(losses):
    return np.mean(losses) + np.std(losses)

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def calculate_sensitivity(predictions, title):
    TP = np.count_nonzero(~predictions)
    FN = np.count_nonzero(predictions)
    sensitivity = 100 * (TP / (TP + FN))
    
    print(f'{title}: {sensitivity:.1f}%')
    return sensitivity

def calculate_specificity(predictions, title):
    TN = np.count_nonzero(predictions)
    FP = np.count_nonzero(~predictions)
    specificity = 100 * (TN / (TN + FP))
    
    print(f'{title}: {specificity:.1f}%')
    return specificity


