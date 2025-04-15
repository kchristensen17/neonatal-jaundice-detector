import streamlit as st
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# Download the model from Google Drive 
GOOGLE_DRIVE_FILE_ID = "1pgnXqziurrN_iBHDQtwKPXG2hMlN92kA"
MODEL_PATH = "jaundice_detection_modelv2.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

download_model()

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("Neonatal Jaundice Detector")
st.write("Upload an image to get a jaundice prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0][0]
        label = "No Jaundice Detected" if prediction >= 0.5 else "Jaundice Detected"

        st.subheader(f"Prediction: {label}")
