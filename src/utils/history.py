import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
