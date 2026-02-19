import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶", layout="centered")

# Title
st.title("🐈‍⬛ 🐕 Cat vs Dog Image classifier")
st.write("Upload an image and the model will predict whether it is a Cat or a Dog.")

# Load Model (Cache for performance)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Cat_dog_model.h5", compile=False)
    return model

model = load_model()

IMG_SIZE = 150

# Iamge Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)


    # Preprocess Image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Prediction 
    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    st.write("### Prediction Result:")

    if confidence > 0.5:
        st.write(f"**Prediction:** Dog (Confidence: {confidence:.2f})")
    else:
        st.write(f"**Prediction:** Cat (Confidence: {1-confidence:.2f})")

    st.write("---------------------")
    st.write("Model Output Value:", confidence)