import tensorflow as tf

from src.models.model import MeiboModel, freeze_batch_norm_layers

input_shape = (224, 448, 3)


class Model6(MeiboModel):
    def __init__(self) -> None:
        base_model = tf.keras.applications.EfficientNetV2B0(
            weights=None, include_top=False, input_shape=input_shape
        )

        base_model.trainable = True
        freeze_batch_norm_layers(base_model)

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)

        super().__init__(model)
