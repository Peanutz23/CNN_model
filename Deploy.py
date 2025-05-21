import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("CNN_model.h5")

st.title("Vehicle Classifier")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized).astype("float32") / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    st.write("Prediction:", np.argmax(prediction))
