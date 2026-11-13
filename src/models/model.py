from __future__ import annotations

from typing import TypeAlias, cast

import tensorflow as tf

from src.utils.logger import Logger

LEARN_RATE = 0.00005


### Training reducing learing rate on  plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,  # New LR = LR * factor (e.g., 0.001 -> 0.0002)
    patience=5,  # Wait 5 epochs of no improvement before dropping LR
    min_lr=1e-6,  # Don't let the LR go below this value
    verbose=1,  # Logs a message when the LR is reduced
)


class MeiboModel:
    type Model = tf.keras.Model[tf.Tensor, tf.Tensor]
    Dataset: TypeAlias = tf.data.Dataset[tuple[tf.Tensor, int]]

    def __init__(self, model: Model) -> None:
        self.model: MeiboModel.Model = model

        self.early_stop: tf.keras.callbacks.EarlyStopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,  # Stop if val_loss doesn't improve for 5 epochs
                restore_best_weights=True,
            )
        )

    def compile(self) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, training_data: Dataset, validation_data: Dataset) -> tf.keras.callbacks.History:
        return self.model.fit(
            x=training_data,
            validation_data=validation_data,
            epochs=5,
            callbacks=[
                self.early_stop,
                Logger().csv_logger,
            ],
        )

        # NOTE: i removed the class weight

    def evaluate(self, test_data: Dataset) -> dict[str, float]:
        return cast(
            dict[str, float],
            cast(object, self.model.evaluate(test_data, return_dict=True)),
        )
        

