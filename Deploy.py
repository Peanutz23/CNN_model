import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("CNN_model.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Vehicle Classifier")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized).astype("float32") / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    
    # Get predicted label
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    
    st.write("Prediction:", predicted_label)
