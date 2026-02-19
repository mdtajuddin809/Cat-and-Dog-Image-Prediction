import tensorflow as tf

model = tf.keras.models.load_model("cat_dog_model_prediction.h5", compile=False)
model.save("cat_dog_model_new.keras")
print("Model saved successfully!")