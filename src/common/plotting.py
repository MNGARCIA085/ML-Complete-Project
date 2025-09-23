import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_plot(fig, path):
    """Save and close a matplotlib figure."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_history_curve(history, key, val_key, model_name, ylabel, title_suffix, filename):
    """Plot training vs validation curve for a given metric."""
    fig, ax = plt.subplots()
    ax.plot(history[key], label=f"train_{key}")
    ax.plot(history[val_key], label=f"val_{key}")
    ax.set_title(f"{model_name} {title_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    path = filename  # stable name
    save_plot(fig, path)
    return path


def plot_confusion_matrix(model, val_ds, model_name, filename):
    """Plot and save confusion matrix from validation dataset."""
    y_true = np.concatenate([y for _, y in val_ds], axis=0)
    y_pred = np.argmax(model.predict(val_ds), axis=1)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")

    path = filename  # stable name
    save_plot(fig, path)
    return path





def plot_comparison(results_list, metric, ylabel, title, filename):
    """Plot comparison of validation metrics across multiple models."""
    fig, ax = plt.subplots()
    for r in results_list:
        h = r["history"].history
        ax.plot(h[f"val_{metric}"], label=f"{r['model_name']} val_{metric}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    save_plot(fig, filename)
    return filename