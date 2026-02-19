import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import zipfile
import tempfile
import os

# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶", layout="centered")

# Title
st.title("🐈‍⬛ 🐕 Cat vs Dog Image Classifier")
st.write("Upload an image and the model will predict whether it is a Cat or a Dog.")

# Load Model with batch_shape compatibility fix
@st.cache_resource
def load_my_model():
    model_path = "cat_dog_model_new.keras"

    try:
        # First, try loading directly
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    except TypeError as e:
        if "batch_shape" not in str(e):
            raise e

        # Patch: replace "batch_shape" with "shape" in the model config
        with zipfile.ZipFile(model_path, "r") as z:
            names = z.namelist()
            files = {name: z.read(name) for name in names}

        # Fix the config.json
        config_str = files["config.json"].decode("utf-8")
        config_str = config_str.replace('"batch_shape"', '"shape"')
        files["config.json"] = config_str.encode("utf-8")

        # Write patched model to a temp file
        patched_path = tempfile.mktemp(suffix=".keras")
        with zipfile.ZipFile(patched_path, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)

        model = tf.keras.models.load_model(patched_path, compile=False)
        os.remove(patched_path)
        return model

model = load_my_model()

IMG_SIZE = 150

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    st.write("### Prediction Result:")
    if confidence > 0.5:
        st.success(f"🐶 **Dog** (Confidence: {confidence:.2f})")
    else:
        st.success(f"🐱 **Cat** (Confidence: {1 - confidence:.2f})")

    st.write("---")
    st.write("Model Output Value:", confidence)