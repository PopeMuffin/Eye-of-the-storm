import tensorflow as tf

from src.models.model import MeiboModel

input_shape = (224, 448, 3)


class Model4(MeiboModel):
    def __init__(self) -> None:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                # --- Block 1 ---
                tf.keras.layers.Conv2D(32, (3, 3), padding="valid"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                # --- Block 2 ---
                tf.keras.layers.Conv2D(64, (3, 3), padding="valid"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                # --- Block 3 ---
                tf.keras.layers.Conv2D(128, (3, 3), padding="valid"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(),
                # --- Block 4 ---
                tf.keras.layers.Conv2D(128, (3, 3), padding="valid"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                # --- The Global Average Pooling Layer ---
                tf.keras.layers.GlobalAveragePooling2D(),
                # --- Fully Connected ---
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.1),  # previously 0.5
                # Output layer for 4 classes (0, 1, 2, 3)
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )

        super().__init__(model)
