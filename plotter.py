import matplotlib.pyplot as plt
import numpy as np
import os


def plot_images(
    dataloader,
    pred_labels,
    class_names,
    title="Classification_results",
    grid_size=(4, 4),
):
    images, labels = next(iter(dataloader))
    images = images.numpy()
    plt.figure(figsize=(8, 8))
    for i in range(grid_size[0] * grid_size[1]):
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # imshow() takes args in order (height, width, color_model), image stores values in order (RGB, height, width)
        img = np.transpose(images[i], (1, 2, 0))
        # scales RGB values to [0, 1] interval which is required by imshow()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.xlabel(class_names[pred_labels[i]])
    if not os.path.exists(f"plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/{title}.png")


def plot_loss(losses, epochs, title="Loss_minimalization"):
    plt.figure(figsize=(8, 8))
    bars = plt.bar(x=list(range(1, epochs + 1)), height=losses, width=0.8)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Losses values")

    for bar in bars:
        y_val = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_val,
            round(y_val, 3),
            ha="center",
            va="bottom",
        )

    if not os.path.exists(f"plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/{title}.png")


def plot_metrics(metrics, title="Metrics"):
    if len(metrics) != 4:
        raise ValueError(
            "Metrics should contain exactly four elements: [Acc, Recall, Precision, F-score]"
        )
    plt.figure(figsize=(6, 6))
    bars = plt.bar(
        x=["Acc", "Recall", "Precision", "F-score"], height=metrics, width=0.8
    )
    plt.title(title)
    plt.xlabel("Metrics types")
    plt.ylabel("Model efficiency")

    for bar in bars:
        y_val = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_val,
            round(y_val, 3),
            ha="center",
            va="bottom",
        )

    if not os.path.exists(f"plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/{title}.png")
