import random
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import var
import Preprocessing as prep
import numpy as np
import os

CHECKPOINT_PATH = r""

positive_dirs = []
negative_dirs = []

random.seed(123)

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

model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)

if __name__ == "__main__":
    wandb.init(
        project="Bumblebees",

        config={
            "layer_1": 128,
            "activation_1": "relu",
            "layer_2": 256,
            "activation_2": "relu",
            "layer_3": 512,
            "activation_3": "relu",
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metric": ['accuracy', 'AUC', "F1Score", 'TruePositives', 'FalsePositives', 'TrueNegatives', 'FalseNegatives'],
            "learning_rate": 1e-2,
            "epoch": 50,
            "batch_size": var.BATCH_SIZE,
        }
    )

    config = wandb.config

    positive_spectrograms = np.concatenate([prep.create_spectrogram_tensors(dir) for dir in positive_dirs], axis=0)
    negative_spectrograms = np.concatenate([prep.create_spectrogram_tensors(dir) for dir in negative_dirs], axis=0)
    positive_elements = int(len(positive_spectrograms) * var.VALIDATION_SPLIT)
    negative_elements = int(len(negative_spectrograms) * var.VALIDATION_SPLIT)

    train_ds = tf.data.Dataset.from_tensor_slices((np.concatenate((positive_spectrograms[:positive_elements], negative_spectrograms[:negative_elements]), axis=0), 
                                                np.concatenate((np.ones(positive_elements), np.zeros(negative_elements)), axis=0)))
    val_ds = tf.data.Dataset.from_tensor_slices((np.concatenate((positive_spectrograms[positive_elements:], negative_spectrograms[negative_elements:]), axis=0), 
                                              np.concatenate((np.ones(len(positive_spectrograms)-positive_elements), np.zeros(len(negative_spectrograms)-negative_elements)), axis=0)))
    train_ds = train_ds.shuffle(4000)
    val_ds = val_ds.shuffle(4000)
    train_ds = train_ds.batch(var.BATCH_SIZE)
    val_ds = val_ds.batch(var.BATCH_SIZE)
    
    print(len(train_ds), len(val_ds))

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_delta=1e-80, min_lr=1e-10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=0.5), tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()]
   )

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=config.epoch,
                        batch_size=var.BATCH_SIZE,
                        callbacks=[
                        WandbMetricsLogger(log_freq=1),
                        # WandbModelCheckpoint(
                        #     CHECKPOINT_PATH, save_weights_only=True),
                        reduce_lr_on_plateau,
                        cp_callback
                        ],
                        )
    wandb.finish()
    # model.save_weights(CHECKPOINT_PATH),
    model.summary()
