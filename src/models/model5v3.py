import tensorflow as tf

from src.models.model import MeiboModel

input_shape = (224, 448, 3)


class Model5v3(MeiboModel):
    def __init__(self) -> None:
        # weights='imagenet' uses pre-trained weights to start with
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

        base_model.trainable = True

        model = tf.keras.models.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),  # Reduces the spatial dimensions
                tf.keras.layers.Dropout(0.2),  # CRITICAL for your overfitting issue
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),  # Helps with training stability
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(4, activation="softmax"),  # Your 4 grading classes
            ]
        )

        super().__init__(model)
