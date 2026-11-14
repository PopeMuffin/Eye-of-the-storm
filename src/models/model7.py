import tensorflow as tf

from src.models.model import MeiboModel

input_shape = (224, 448, 3)


def _gland_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    # Branch 1: Small details (3x3)
    a = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    # Branch 2: Wider context (5x5)
    b = tf.keras.layers.Conv2D(filters, (5, 5), padding="same", activation="relu")(x)

    x = tf.keras.layers.Concatenate()([a, b])
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.MaxPooling2D((2, 2))(x)


class Model7(MeiboModel):
    def __init__(self) -> None:
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Initial high-res feature extraction
        x = tf.keras.layers.Conv2D(32, (7, 7), strides=2, padding="same", activation="relu")(
            inputs
        )

        # Blocks that preserve texture
        x = _gland_block(x, 32)
        x = _gland_block(x, 64)
        x = _gland_block(x, 128)

        # The "Decision" Head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Better for "percentage of loss"
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)

        super().__init__(model)
