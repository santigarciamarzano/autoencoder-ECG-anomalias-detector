import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    np.random.seed(23)
    tf.random.set_seed(23)
    
    entrada = Input(shape=(input_dim,))
    
    encoder = Dense(32, activation='relu')(entrada)
    encoder = Dense(16, activation='relu')(encoder)
    encoder = Dense(8, activation='relu')(encoder)
   
    decoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    autoencoder = Model(inputs=entrada, outputs=decoder) # entrada + decoder (que ya contiene el encoder)
    
    return autoencoder

def compile_and_train(autoencoder, x_train, x_test, epochs=20, batch_size=512):
    autoencoder.compile(optimizer='adam', loss='mae')
    historia = autoencoder.fit(x_train, x_train, 
                               epochs=epochs, 
                               batch_size=batch_size,
                               validation_data=(x_test, x_test),
                               shuffle=True)
    return historia

def load_autoencoder_weights(model, weights_path):
    model.load_weights(weights_path)
    return model
