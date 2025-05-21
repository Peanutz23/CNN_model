import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# load the model for the app
model = tf.keras.models.load_model('vehicle_classifier.h5')

# classify labels of the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Vehicle Image Classifier")
st.write("Upload an image and I'll try to classify it!")

# user image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # process the image for classifying and predicting
    image = image.resize((32, 32))  # Resize to match model input
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 32, 32, 3)

    # predictions of the model
    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: `{predicted_label}`")
