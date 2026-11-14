import tensorflow as tf

from src.models.model import MeiboModel, freeze_batch_norm_layers

input_shape = (224, 448, 3)


class Model6vclaude(MeiboModel):
    def __init__(self, unfreeze_from_layer: int | None = None) -> None:
        base_model = tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",  # Always start with imagenet, never None
            include_top=False,
            input_shape=input_shape,  # Square input - fixes aspect ratio problem
        )

        # Re-initialise ONLY the first conv layer - it needs to adapt from
        # RGB to replicated greyscale
        first_conv = next(
            layer
            for layer in base_model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        )
        new_kernel = tf.keras.initializers.HeNormal()(shape=first_conv.kernel.shape)
        first_conv.kernel.assign(new_kernel)
        if first_conv.bias is not None:
            first_conv.bias.assign(tf.zeros_like(first_conv.bias))

        # Default: freeze everything
        base_model.trainable = False

        # Stage 2+: selectively unfreeze from a layer index onward
        if unfreeze_from_layer is not None:
            for layer in base_model.layers[unfreeze_from_layer:]:
                layer.trainable = True

        # Always keep first conv trainable - must adapt to greyscale
        first_conv.trainable = True

        # Always freeze ALL BatchNorm layers - critical for small datasets
        freeze_batch_norm_layers(base_model)

        inputs = tf.keras.layers.Input(shape=input_shape)

        # training=False is critical - keeps BN running in inference mode
        # even during training, preventing frozen BN from corrupting gradients
        x = base_model(inputs, training=False)

        # Dual pooling - avg captures distributed texture, max captures peak
        # activations, both useful for grading tasks
        avg_p = tf.keras.layers.GlobalAveragePooling2D()(x)
        max_p = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Concatenate()([avg_p, max_p])

        x = tf.keras.layers.BatchNormalization()(x)  # head BN, always trainable
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)

        super().__init__(model)
