"""
Training Engine for Car Spare Parts Classification CNN.
Supports:
- AdamW optimizer with configurable learning rate & weight decay
- ReduceLROnPlateau learning-rate scheduling
- Early stopping based on validation loss/accuracy
- Checkpointing of the best performing model
- Real-time display of epoch loss and accuracy
"""
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from .config import (
        DEVICE, RANDOM_SEED, NUM_EPOCHS, LEARNING_RATE,
        WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, XCEPTION_MODEL_PATH,
        MODELS_DIR
    )
    from .models import XceptionSparePartsModel, get_model_summary
except ImportError:
    from config import (
        DEVICE, RANDOM_SEED, NUM_EPOCHS, LEARNING_RATE,
        WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, XCEPTION_MODEL_PATH,
        MODELS_DIR
    )
    from models import XceptionSparePartsModel, get_model_summary


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed across all libraries for deterministic reproducibility and configure CPU threading."""
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        cpu_cores = os.cpu_count() or 8
        torch.set_num_threads(cpu_cores)


class EarlyStopping:
    """Early stops training when validation loss fails to improve after patience epochs."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Execute single training epoch and return (epoch_loss, epoch_acc)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation loader without gradient updates."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    save_path: Path = XCEPTION_MODEL_PATH,
    model_name: str = "Xception"
) -> Dict[str, Any]:
    """Complete training loop with learning rate scheduling, early stopping, and checkpointing."""
    set_seed(RANDOM_SEED)
    device = DEVICE
    model = model.to(device)

    # Class-weighted CrossEntropyLoss
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "epoch_times": []
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    print("\n" + "=" * 75)
    print(f"  TRAINING PIPELINE: {model_name}")
    print(f"  Target Device: {device} | Max Epochs: {num_epochs} | Initial LR: {learning_rate}")
    print("=" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'LR':<9} | {'Time'}")
    print("-" * 75)

    start_training_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(round(epoch_duration, 2))

        # Checkpointing if validation accuracy improves or loss improves
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "model_summary": get_model_summary(model)
            }, save_path)
            saved_indicator = " * [BEST SAVED]"
        else:
            saved_indicator = ""

        print(f"{epoch:>3}/{num_epochs:<3} | {train_loss:>10.4f} | {train_acc * 100:>8.2f}% | {val_loss:>9.4f} | {val_acc * 100:>7.2f}% | {current_lr:>8.1e} | {epoch_duration:>5.1f}s{saved_indicator}")

        # Check Early Stopping
        if early_stopping.step(val_loss):
            print(f"\n[INFO] Early stopping triggered at epoch {epoch} (no validation loss improvement for {patience} epochs).")
            break

    total_training_time = time.time() - start_training_time
    print("-" * 75)
    print(f"Training Complete in {total_training_time / 60:.2f} minutes.")
    print(f"Best Model Checkpoint: Epoch {best_epoch} with Val Acc = {best_val_acc * 100:.2f}%, Val Loss = {best_val_loss:.4f}")
    print(f"Model checkpoint saved to: {save_path}\n")

    # Load best checkpoint weights before returning
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_training_time": round(total_training_time, 2),
        "epochs_trained": len(history["train_loss"])
    }
