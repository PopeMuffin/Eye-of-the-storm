import tensorflow as tf

from src.models.model import MeiboModel, freeze_batch_norm_layers

input_shape = (224, 448, 3)


class Model10(MeiboModel):
    def __init__(self) -> None:
        # DenseNet-121 is deeper but has fewer parameters than ResNet50
        base_model = tf.keras.applications.DenseNet121(
            weights=None,  # Or 'imagenet' if you want to try transfer learning
            include_top=False,
            input_shape=input_shape,
        )

        # For medical masks, training from scratch (trainable=True) is often
        # better than ImageNet weights because masks don't look like
        # "natural" objects.
        base_model.trainable = True

        # DenseNet relies HEAVILY on BN layers. If your batch size is small
        # (e.g., < 16), keep these frozen to use the global statistics and
        # prevent training instability.
        freeze_batch_norm_layers(base_model)

        # Note: DenseNet's output is naturally "dense" in features, so we
        # use a robust head to map to the 4 meiboscore classes.
        model = tf.keras.models.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.BatchNormalization(),
                # First Dense Block
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),  # High dropout for the large feature set
                # Second Dense Block
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                # 4 classes: Grade 0, 1, 2, 3
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )

        super().__init__(model)
