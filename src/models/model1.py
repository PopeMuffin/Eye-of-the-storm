from src.models.model import MeiboModel

import tensorflow as tf


input_shape = (224, 448, 3)


class Model1(MeiboModel):
    def __init__(self) -> None:
        model = tf.keras.models.Sequential(
            [
                # First Convolutional Block
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                # Second Convolutional Block
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                # Third Convolutional Block
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                # Flatten and Dense layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),  # Helps prevent overfitting
                # Output layer: 4 neurons for classes 0, 1, 2, 3
                tf.keras.layers.Dense(4, activation="softmax"),  # original
                # layers.Dense(3)  # coral method with changed labels from scalar to vector
                # layers.Dense(1, activation='sigmoid') # use when we use the meanabsolute error as loss function
            ]
        )

        super().__init__(model)
