import tensorflow as tf

from src.models.model import MeiboModel, freeze_batch_norm_layers

input_shape = (224, 448, 3)


class Model9(MeiboModel):
    def __init__(self) -> None:
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

        # Don't freeze the base model - prevents the pre-trained weights
        # from being destroyed early on
        base_model.trainable = True
        freeze_batch_norm_layers(base_model)

        model = tf.keras.models.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),  # Reduces the spatial dimensions
                tf.keras.layers.BatchNormalization(),  # Helps with training stability
                tf.keras.layers.Dropout(0.5),  # CRITICAL for your overfitting issue
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(4, activation="softmax"),  # Your 4 grading classes
            ]
        )

        super().__init__(model)
