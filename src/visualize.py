"""
Visualization Module for Car Spare Parts Classification Pipeline.
Generates publication-quality charts and plots for the project report:
1. Training vs. Validation Accuracy Curve
2. Training vs. Validation Loss Curve
3. Confusion Matrix Heatmap
4. Sample Predictions Grid
5. Correctly Classified Examples
6. Incorrectly Classified Examples
"""
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .config import (
        ACCURACY_CURVE_PATH, LOSS_CURVE_PATH, CONFUSION_MATRIX_PATH,
        SAMPLE_PREDICTIONS_PATH, CORRECT_PREDICTIONS_PATH, INCORRECT_PREDICTIONS_PATH
    )
except ImportError:
    from config import (
        ACCURACY_CURVE_PATH, LOSS_CURVE_PATH, CONFUSION_MATRIX_PATH,
        SAMPLE_PREDICTIONS_PATH, CORRECT_PREDICTIONS_PATH, INCORRECT_PREDICTIONS_PATH
    )


# Set clean modern aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})


def plot_accuracy_curve(history: Dict[str, List[float]], save_path: Path = ACCURACY_CURVE_PATH):
    """Plot and save training vs validation accuracy progression across epochs."""
    epochs = range(1, len(history["train_acc"]) + 1)
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc = [a * 100 for a in history["val_acc"]]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.plot(epochs, train_acc, label="Training Accuracy", color="#2563eb", linewidth=2.4, marker="o", markersize=4)
    ax.plot(epochs, val_acc, label="Validation Accuracy", color="#10b981", linewidth=2.4, marker="s", markersize=4)

    best_epoch = int(np.argmax(val_acc)) + 1
    best_acc = max(val_acc)
    ax.scatter([best_epoch], [best_acc], color="#ef4444", s=100, zorder=5, label=f"Best Val: {best_acc:.1f}% (Epoch {best_epoch})")
    ax.annotate(
        f"Peak: {best_acc:.1f}%",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 0.5, best_acc - 3),
        arrowprops=dict(facecolor="#ef4444", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#b91c1c"
    )

    ax.set_title("Training vs. Validation Accuracy Across Epochs", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch Number", fontsize=11, fontweight="medium")
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="medium")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_loss_curve(history: Dict[str, List[float]], save_path: Path = LOSS_CURVE_PATH):
    """Plot and save training vs validation loss progression across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.plot(epochs, train_loss, label="Training Loss", color="#d97706", linewidth=2.4, marker="o", markersize=4)
    ax.plot(epochs, val_loss, label="Validation Loss", color="#dc2626", linewidth=2.4, marker="^", markersize=4)

    best_epoch = int(np.argmin(val_loss)) + 1
    min_loss = min(val_loss)
    ax.scatter([best_epoch], [min_loss], color="#059669", s=100, zorder=5, label=f"Min Val Loss: {min_loss:.4f} (Epoch {best_epoch})")

    ax.set_title("Training vs. Validation Loss Across Epochs", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch Number", fontsize=11, fontweight="medium")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="medium")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path = CONFUSION_MATRIX_PATH
):
    """Plot and save normalized confusion matrix heatmap."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    
    # Format labels: count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    display_classes = [c.replace("_", " ").title() for c in class_names]
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=display_classes,
        yticklabels=display_classes,
        cbar_kws={"label": "Normalized Recall Proportion"},
        ax=ax,
        linewidths=0.5,
        linecolor="#f1f5f9"
    )

    ax.set_title("Spare Parts Recognition Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Spare Part", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual Spare Part (Ground Truth)", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_prediction_grid(
    paths: List[str],
    true_labels: List[int],
    pred_labels: List[int],
    probs: List[List[float]],
    class_names: List[str],
    save_path: Path,
    title: str,
    filter_mode: str = "all",  # "all", "correct", "incorrect"
    max_samples: int = 12
):
    """Render a visual grid of spare-part images with ground truth, prediction, and confidence."""
    indices = []
    for idx in range(len(paths)):
        is_corr = (true_labels[idx] == pred_labels[idx])
        if filter_mode == "correct" and is_corr:
            indices.append(idx)
        elif filter_mode == "incorrect" and not is_corr:
            indices.append(idx)
        elif filter_mode == "all":
            indices.append(idx)

    if not indices:
        # Fallback dummy figure if e.g. 0 errors
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No instances found for filter: {filter_mode}", ha="center", va="center")
        ax.axis("off")
        plt.savefig(save_path)
        plt.close()
        return

    # Deterministically select up to max_samples
    np.random.seed(42)
    selected_indices = np.random.choice(indices, size=min(len(indices), max_samples), replace=False)

    num_samples = len(selected_indices)
    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), dpi=250)
    axes = np.array(axes).reshape(-1)

    for i, sel_idx in enumerate(selected_indices):
        ax = axes[i]
        img_path = paths[sel_idx]
        t_cls = class_names[true_labels[sel_idx]].replace("_", " ").title()
        p_cls = class_names[pred_labels[sel_idx]].replace("_", " ").title()
        conf = probs[sel_idx][pred_labels[sel_idx]] * 100

        try:
            with Image.open(img_path) as img:
                ax.imshow(img.convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "Image Unreadable", ha="center")

        is_correct = (true_labels[sel_idx] == pred_labels[sel_idx])
        color = "#15803d" if is_correct else "#b91c1c"
        status_symbol = "[OK]" if is_correct else "[X]"

        ax.set_title(
            f"{status_symbol} Pred: {p_cls} ({conf:.1f}%)\nTrue: {t_cls}",
            fontsize=9,
            fontweight="bold",
            color=color,
            pad=6
        )
        ax.axis("off")

    # Hide remaining empty subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


from typing import Dict, List, Any, Optional

def generate_all_visualizations(
    history: Dict[str, Any],
    eval_results: Dict[str, Any],
    class_names: List[str],
    out_dir: Optional[Path] = None
):
    """Generate and save all 6 required report visualizations."""
    
    if out_dir is not None:
        acc_path = out_dir / "accuracy_curve.png"
        loss_path = out_dir / "loss_curve.png"
        cm_path = out_dir / "confusion_matrix.png"
        samp_path = out_dir / "sample_predictions.png"
        corr_path = out_dir / "correct_predictions.png"
        incorr_path = out_dir / "incorrect_predictions.png"
    else:
        acc_path = ACCURACY_CURVE_PATH
        loss_path = LOSS_CURVE_PATH
        cm_path = CONFUSION_MATRIX_PATH
        samp_path = SAMPLE_PREDICTIONS_PATH
        corr_path = CORRECT_PREDICTIONS_PATH
        incorr_path = INCORRECT_PREDICTIONS_PATH

    print("[VISUALIZATION] Generating training accuracy curves...")
    plot_accuracy_curve(history, acc_path)

    print("[VISUALIZATION] Generating training loss curves...")
    plot_loss_curve(history, loss_path)

    print("[VISUALIZATION] Generating confusion matrix heatmap...")
    cm = np.array(eval_results["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, cm_path)

    paths = eval_results["all_paths"]
    true_labels = eval_results["all_labels"]
    pred_labels = eval_results["all_preds"]
    probs = eval_results["all_probs"]

    print("[VISUALIZATION] Generating sample test predictions grid...")
    plot_prediction_grid(
        paths, true_labels, pred_labels, probs, class_names,
        samp_path,
        title="Sample Test Predictions: Car Spare Parts CNN",
        filter_mode="all",
        max_samples=12
    )

    print("[VISUALIZATION] Generating correctly classified examples...")
    plot_prediction_grid(
        paths, true_labels, pred_labels, probs, class_names,
        corr_path,
        title="Correctly Classified Test Examples (High Confidence)",
        filter_mode="correct",
        max_samples=12
    )

    print("[VISUALIZATION] Generating incorrectly classified examples...")
    plot_prediction_grid(
        paths, true_labels, pred_labels, probs, class_names,
        incorr_path,
        title="Misclassified Test Examples (Error Diagnostics)",
        filter_mode="incorrect",
        max_samples=12
    )
    print("[VISUALIZATION] All 6 visualization figures generated successfully!")
