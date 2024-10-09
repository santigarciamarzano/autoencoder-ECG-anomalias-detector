import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scripts.model import build_autoencoder, load_autoencoder_weights
from scripts.preprocessing import preprocess_data
from scripts.evaluation import calculate_threshold, predict, calculate_sensitivity, calculate_specificity

st.title("Autoencoder para Detección de Anomalías")

input_dim = 140  
autoencoder = build_autoencoder(input_dim)
autoencoder = load_autoencoder_weights(autoencoder, 'models/autoencoder_weights.weights.h5')


uploaded_file = st.file_uploader("Cargar archivo de datos de prueba", type=["csv"])
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file, header=None)

    _, x_test_scaled, scaler = preprocess_data(train_data=None, test_data=test_data)

    reconstructions = autoencoder.predict(x_test_scaled)
    loss = tf.keras.losses.mae(reconstructions, x_test_scaled)
    threshold = calculate_threshold(loss)
    predictions = predict(autoencoder, x_test_scaled, threshold)

    
    st.write(f"Umbral calculado: {threshold:.4f}")
    st.write(f"Predicciones: {predictions.numpy()}")

    sensitivity = calculate_sensitivity(predictions, "Sensitividad (anormales)")
    specificity = calculate_specificity(predictions, "Especificidad (normales)")
    
    st.write(f"Sensitividad: {sensitivity:.2f}%")
    st.write(f"Especificidad: {specificity:.2f}%")



