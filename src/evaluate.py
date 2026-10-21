"""
Evaluation module for Car Spare Parts Classification CNN.
Evaluates the model on unseen test data, computing accuracy, loss, precision,
recall, F1-score, full classification report, confusion matrix, and top confused pairs.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

try:
    from .config import (
        DEVICE, EVALUATION_METRICS_FILE, CLASSIFICATION_REPORT_FILE
    )
except ImportError:
    from config import (
        DEVICE, EVALUATION_METRICS_FILE, CLASSIFICATION_REPORT_FILE
    )


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    criterion: Optional[nn.Module] = None,
    device: torch.device = DEVICE,
    metrics_file: Path = EVALUATION_METRICS_FILE,
    report_file: Path = CLASSIFICATION_REPORT_FILE
) -> Dict[str, Any]:
    """Perform rigorous evaluation on the unseen test split."""
    model.eval()
    model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    total_samples = len(all_labels)
    test_loss = running_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_accuracy = float(np.mean(all_preds == all_labels))

    # Overall macro and weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0  # type: ignore
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0  # type: ignore
    )

    # Scikit-learn classification report
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0  # type: ignore
    )
    report_text = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0  # type: ignore
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Identify most frequently confused pairs
    confused_pairs = []
    num_classes = len(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "actual": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j])
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    results = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": confused_pairs[:5],
        "all_preds": all_preds.tolist(),
        "all_labels": all_labels.tolist(),
        "all_probs": all_probs.tolist(),
        "all_paths": all_paths
    }

    # Save to outputs directory
    with open(metrics_file, "w", encoding="utf-8") as f:
        # Don't serialize giant arrays in the summary metrics json
        summary_to_save = {k: v for k, v in results.items() if not k.startswith("all_")}
        json.dump(summary_to_save, f, indent=2)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("CAR SPARE PARTS CLASSIFICATION REPORT (UNSEEN TEST SET)\n")
        f.write("=" * 65 + "\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Macro F1:      {f1_macro:.4f}\n")
        f.write(f"Weighted F1:   {f1_weighted:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write(str(report_text) + "\n")
        f.write("\nTOP CONFUSED SPARE PART PAIRS:\n")
        if confused_pairs:
            for pair in confused_pairs[:5]:
                f.write(f"  • Actual '{pair['actual']}' confused as '{pair['predicted']}': {pair['count']} instances\n")
        else:
            f.write("  • Perfect classification, zero confusion instances!\n")

    return results


def print_evaluation_summary(results: Dict[str, Any], class_names: List[str]) -> None:
    """Print clean evaluation metrics summary."""
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON UNSEEN TEST SET")
    print("=" * 65)
    print(f"Final test accuracy:  {results['test_accuracy'] * 100:.2f}%")
    print(f"Test loss:            {results['test_loss']:.4f}")
    print(f"Precision (Macro):    {results['precision_macro']:.4f}  | (Weighted): {results['precision_weighted']:.4f}")
    print(f"Recall (Macro):       {results['recall_macro']:.4f}  | (Weighted): {results['recall_weighted']:.4f}")
    print(f"F1-score (Macro):     {results['f1_macro']:.4f}  | (Weighted): {results['f1_weighted']:.4f}")
    print("-" * 65)
    print("Per-Class Performance:")
    for cls in class_names:
        c_metrics = results["classification_report"].get(cls, {})
        p = c_metrics.get("precision", 0.0) * 100
        r = c_metrics.get("recall", 0.0) * 100
        f = c_metrics.get("f1-score", 0.0) * 100
        sup = int(c_metrics.get("support", 0))
        print(f"  - {cls:<16}: Prec: {p:>5.1f}% | Rec: {r:>5.1f}% | F1: {f:>5.1f}% | Support: {sup:>3}")
    print("-" * 65)
    print("Top Confused Spare Part Pairs:")
    if results.get("top_confused_pairs"):
        for pair in results["top_confused_pairs"][:5]:
            print(f"  - Actual '{pair['actual']}' misclassified as '{pair['predicted']}': {pair['count']} times")
    else:
        print("  - No misclassifications detected!")
    print("=" * 65 + "\n")
