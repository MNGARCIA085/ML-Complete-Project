import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc


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


def plot_confusion_matrix(model, ds, model_name, filename):
    """Plot and save confusion matrix from a Keras model and tf.data.Dataset."""
    y_true = np.concatenate([y for _, y in ds], axis=0)
    y_pred_probs = model.predict(ds)

    # Binary vs Multiclass
    if y_pred_probs.ndim == 1 or y_pred_probs.shape[1] == 1:
        # Binary sigmoid output
        y_pred = (y_pred_probs.ravel() > 0.5).astype(int)
    else:
        # Multiclass softmax
        y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    print(cm)

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





def plot_roc_curve(model, ds, model_name, filename):
    """Plot and save ROC curve from validation dataset."""
    y_true = np.concatenate([y for _, y in ds], axis=0)
    y_score = model.predict(ds)  # raw probabilities

    fpr, tpr, _ = roc_curve(y_true, y_score.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})")
    

    # Common settings
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curve")
    ax.legend(loc="lower right")

    path = filename
    save_plot(fig, path)
    return path






