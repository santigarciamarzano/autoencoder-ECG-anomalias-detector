import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def plot_ecg_samples(x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, ind=10):
    normal = x_train_1[ind]
    anormal_2 = x_train_2[ind]
    anormal_3 = x_train_3[ind]
    anormal_4 = x_train_4[ind]
    anormal_5 = x_train_5[ind]

    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(140), normal)
    plt.plot(np.arange(140), anormal_2, 'r--')
    plt.title('Normal vs Anormal 2')

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(140), normal)
    plt.plot(np.arange(140), anormal_3, 'r--')
    plt.title('Normal vs Anormal 3')

    plt.subplot(2, 2, 3)
    plt.plot(np.arange(140), normal)
    plt.plot(np.arange(140), anormal_4, 'r--')
    plt.title('Normal vs Anormal 4')

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(140), normal)
    plt.plot(np.arange(140), anormal_5, 'r--')
    plt.title('Normal vs Anormal 5')

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history["loss"], label="Pérdida set entrenamiento")
    plt.plot(history.history["val_loss"], label="Pérdida set prueba")
    plt.legend()
    plt.show()


def plot_ecg_reconstruction(autoencoder, x_test_normal, x_test_anormal, index=0):
    
    rec_normal = autoencoder.predict(x_test_normal)
    rec_anormal = autoencoder.predict(x_test_anormal)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_test_normal[index], 'b')
    plt.plot(rec_normal[index], 'r')
    plt.fill_between(np.arange(len(x_test_normal[index])), rec_normal[index], x_test_normal[index], color='lightcoral')
    plt.legend(labels=["Original normal", "Reconstruction", "Error"])
    plt.title("Reconstruction of Normal ECG")

    plt.subplot(1, 2, 2)
    plt.plot(x_test_anormal[index], 'b')
    plt.plot(rec_anormal[index], 'r')
    plt.fill_between(np.arange(len(x_test_anormal[index])), rec_anormal[index], x_test_anormal[index], color='lightcoral')
    plt.legend(labels=["Original anormal", "Reconstruction", "Error"])
    plt.title("Reconstruction of Abnormal ECG")

    plt.show()

"""def plot_reconstruction(autoencoder, x_test_1_s, x_test_5_s, dato=5):
    rec_normal = autoencoder.predict(x_test_1_s)
    rec_anormal = autoencoder.predict(x_test_5_s)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_test_1_s[dato], 'b')
    plt.plot(rec_normal[dato], 'r')
    plt.fill_between(np.arange(140), rec_normal[dato], x_test_1_s[dato], color='lightcoral')
    plt.legend(labels=["Original normal", "Reconstruction", "Error"])

    plt.subplot(1, 2, 2)
    plt.plot(x_test_5_s[dato], 'b')
    plt.plot(rec_anormal[dato], 'r')
    plt.fill_between(np.arange(140), rec_anormal[dato], x_test_5_s[dato], color='lightcoral')
    plt.legend(labels=["Original anormal", "Reconstruction", "Error"])

    plt.show()"""


def load_autoencoder_weights(autoencoder, weights_path):
    autoencoder.load_weights(weights_path)
    return autoencoder

def calculate_losses(autoencoder, x_tests):
    losses = [tf.keras.losses.mae(autoencoder.predict(x_test), x_test) for x_test in x_tests]
    return losses

def plot_loss_histograms(losses, threshold=0.08):
    plt.figure(figsize=(15, 8))
    colors = ['#1f77b4', '#ff521b', '#020122', '#eefc57', 'r']
    labels = ['normales (1)', 'anormales (2)', 'anormales (3)', 'anormales (4)', 'anormales (5)']
    
    for i, loss in enumerate(losses):
        plt.hist(loss[None, :], bins=100, alpha=0.75, color=colors[i], label=labels[i])
    
    plt.xlabel('Pérdidas (MAE)')
    plt.ylabel('Nro. ejemplos')
    plt.legend(loc='upper right')
    max_y = max([max(np.histogram(loss[None, :], bins=100)[0]) for loss in losses])
    plt.vlines(threshold, 0, max_y, 'k')
    plt.show()
