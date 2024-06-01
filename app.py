import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load h5 model
model = keras.models.load_model('keras_model.h5')

# Load Labels
with open('labels.txt', 'r') as f:
    labels = f.read().split('\n')

# Input Image
uploaded_file = st.file_uploader("Pilih gambar cuy", type='jpg')
if uploaded_file is not None:
    # Read and preprocess image
    image = tf.io.decode_image(uploaded_file.getvalue(), channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    
    st.image(uploaded_file, caption='Upload cuy', use_column_width=True)

    # Predict
    prediction = model.predict(image)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
