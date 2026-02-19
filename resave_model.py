import tensorflow as tf

print("Loading old model...")
model = tf.keras.models.load_model(
    "cat_dog_model_prediction.h5",
    compile=False
)
print("Old model loaded!")

model.save("cat_dog_model_new.keras")
print("New model saved successfully!")