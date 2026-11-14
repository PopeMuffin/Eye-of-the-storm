import tensorflow as tf

from src.models.model import MeiboModel

input_shape = (224, 448, 3)


def _unet_encoder_block(x: tf.Tensor, filters: int) -> tuple[tf.Tensor, tf.Tensor]:
    # Two convs to refine the fine structures
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    # Save this for a skip connection if you go full U-Net
    pool = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x, pool


class Model8(MeiboModel):
    def __init__(self) -> None:
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Level 1: 64 filters (Captures tiny gland edges)
        f1, p1 = _unet_encoder_block(inputs, 64)

        # Level 2: 128 filters (Captures gland segments)
        f2, p2 = _unet_encoder_block(p1, 128)

        # Level 3: 256 filters (Captures regional atrophy)
        f3, _p3 = _unet_encoder_block(p2, 256)

        # Instead of decoding, we "Fuse" these levels for Scoring
        # This is called a "Multi-scale Fusion"
        gap1 = tf.keras.layers.GlobalAveragePooling2D()(f1)
        gap2 = tf.keras.layers.GlobalAveragePooling2D()(f2)
        gap3 = tf.keras.layers.GlobalAveragePooling2D()(f3)

        combined = tf.keras.layers.Concatenate()([gap1, gap2, gap3])

        x = tf.keras.layers.Dense(512, activation="relu")(combined)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)

        super().__init__(model)
