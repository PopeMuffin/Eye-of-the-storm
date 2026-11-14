import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.utils.logger import Logger

OUTPUT_PATH = "out"


def show_history(history: tf.keras.callbacks.History) -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    _ = plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training Acc")
    plt.plot(epochs, val_acc, "go-", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training Loss")
    plt.plot(epochs, val_loss, "go-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "training_hist.png"))


def plot_confusion_matrix(y_true: list[int], y_pred: list[int]) -> None:
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Grade 0", "Grade 1", "Grade 2", "Grade 3"]

    # 3. Plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix: Meibomian Gland Grading")
    plt.ylabel("Actual Grade")
    plt.xlabel("Predicted Grade")
    plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"))


def log_evaluation(
    history: tf.keras.callbacks.History,
    y_true: list[int],
    y_pred: list[int],
    results: dict[str, float],
    batch_size: int,
    learn_rate: float,
) -> None:
    # sklearn's metric functions have no type annotations, so pyright infers
    # overly narrow parameter/return types from their default values.
    precisions = cast(
        "np.ndarray",
        precision_score(y_true, y_pred, average=None),  # pyright: ignore[reportArgumentType]
    )
    recalls = cast(
        "np.ndarray",
        recall_score(y_true, y_pred, average=None),  # pyright: ignore[reportArgumentType]
    )
    report = cast(str, classification_report(y_true, y_pred))

    unique, counts = np.unique(y_pred, return_counts=True)

    lines = [
        f"\nbatch_size : {batch_size}",
        f"l_rate : {learn_rate}",
        "\nModel Evaluation Results:",
        f"Test Loss: {results['loss']:.4f}",
        f"Test Accuracy: {results['accuracy']:.4f}",
        f"Unique predictions made by model: {unique.tolist()}",
        f"Frequency of each prediction: {counts.tolist()}",
    ]
    for i, (precision, recall) in enumerate(zip(precisions, recalls)):
        lines.append(f"Class {i}: Precision = {precision:.4f}, Recall = {recall:.4f}")
    lines.append(report)
    lines.append(f"\nTraining Accuracy: \n {history.history['accuracy']}")
    lines.append(f"\nValidation Accuracy: \n {history.history['val_accuracy']}")
    lines.append(f"\nTraining loss: \n {history.history['loss']}")
    lines.append(f"\nValidation loss: \n {history.history['val_loss']}")
    lines.append(f"\nArray of Model Predictions (Meiboscores):\n {y_pred}")

    Logger().write("\n".join(lines) + "\n")
