from pathlib import Path
from typing import Self

import tensorflow as tf

LOG_FILENAME = "Details.txt"


class Logger:
    _instance: Self | None = None
    csv_logger: tf.keras.callbacks.CSVLogger

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "csv_logger"):
            return

        _ = Path(LOG_FILENAME).write_text("")
        self.csv_logger = tf.keras.callbacks.CSVLogger(
            LOG_FILENAME, separator="\t", append=True
        )

    def write(self, text: str) -> None:
        with open(LOG_FILENAME, "a") as f:
            _ = f.write(text)
