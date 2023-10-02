import tensorflow as tf
import var

# Initialize Model  => Import Parameters    => Predict Function

CHECKPOINT_PATH = r"./Models/Z1"

print("Using model:", CHECKPOINT_PATH)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 1)),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.build((1, var.IMG_HEIGHT, var.IMG_WIDTH, 1))
model.load_weights(CHECKPOINT_PATH).expect_partial()

def predict(item, confidence=False):
    # Model returns probability of detecting a bumblebee
    if confidence:
        return model.predict(item, verbose=False)
    if model.predict(item, verbose=False) > var.THRESHOLD:
        return 1
    return 0
