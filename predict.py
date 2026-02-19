import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


IMG_SIZE = 150

# Load Model
model = tf.keras.models.load_model("cat_dog_model_prediction.h5")

# Load image 
img_path = r"C:\Users\siraj\Desktop\CNN\datasets\validation\dog\dog 2.jpg"
img = image.load_img(img_path, target_size = (IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis = 0)
img_array = img_array / 255.0

# Prediction 
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("The image is a dog.")
else:
    print("The image is a cat.")
