# AI-Integrated Marketplace for Car Spare Parts Recognition Using an Optimized CNN

An end-to-end, production-grade deep learning pipeline designed to classify automotive spare parts from camera and smartphone imagery. Built with an optimized Convolutional Neural Network (CNN) incorporating residual connections, progressive channel scaling, and Global Average Pooling (GAP), the system is specifically structured to power the recognition engine of an intelligent e-commerce spare parts marketplace.

---

## 1. Project Overview

Finding exact car spare parts can be challenging for drivers, mechanics, and parts suppliers. This project develops an intelligent visual recognition pipeline that:
1. **Automatically ingests and verifies** car spare parts imagery across 10 core automotive categories.
2. **Preprocesses and augments** image data while strictly preserving component integrity.
3. **Trains an Optimized CNN** engineered for high accuracy, minimal parameter footprint (~1.97M parameters), and rapid inference (<10ms).
4. **Benchmarks against a Baseline CNN** to quantify architectural gains.
5. **Evaluates performance on an unseen test set** with per-class precision, recall, F1-score, confusion matrix, and misclassification diagnostics.
6. **Bridges visual AI to marketplace commerce**, mapping classified parts directly to catalog metadata (OEM brands, vehicle compatibility, price ranges, warranty, and verified suppliers).

---

## 2. Dataset Architecture

The pipeline dynamically inspects the dataset folder without hardcoded assumptions.

### Dataset Profile
* **Dataset Location**: `dataset/`
* **Total Valid Images**: 3,439 images
* **Corrupted / Unreadable**: 0 files
* **Duplicate Detection**: 55 intra-class duplicates identified
* **Input Image Resolutions**: Average 240×234 px (ranging from 74×100 to 4272×2848 px)
* **Target Resolution**: $224 \times 224 \times 3$

### Detected Spare Part Classes (10 Categories)
| # | Category | Count | Class Share | Automotive Function |
|---|----------|-------|-------------|---------------------|
| 1 | `battery` | 359 | 10.4% | 12V AGM / Lead-Acid Electrical Storage |
| 2 | `brake_calliper` | 353 | 10.3% | Hydraulic Disc Clamping Assembly |
| 3 | `brake_pad` | 381 | 11.1% | Ceramic / Semi-Metallic Friction Linings |
| 4 | `carburetor` | 389 | 11.3% | Fuel / Air Mechanical Metering |
| 5 | `gas_cap` | 256 | 7.4% | Fuel Tank Seal & Vapor Pressure Valve |
| 6 | `muffler` | 349 | 10.1% | Exhaust Acoustic Attenuator & Backpressure |
| 7 | `oil_filter` | 227 | 6.6% | Synthetic Engine Lubrication Filtration |
| 8 | `radiator` | 343 | 10.0% | Brazed Aluminum Heat Dissipation Core |
| 9 | `spark_plug` | 370 | 10.8% | High-Voltage Iridium / Platinum Igniter |
| 10 | `wheel_rim` | 412 | 12.0% | Alloy / Forged Wheel Assembly |

### Stratified Data Split (Seed 42)
* **Training Set (70%)**: 2,404 images
* **Validation Set (15%)**: 511 images
* **Unseen Test Set (15%)**: 524 images
* **Split Metadata**: Saved reproducibly in `data/train_val_test_split.json`

---

## 3. Preprocessing & Data Augmentation

### Normalization
* Standard channel-wise ImageNet statistics:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### Augmentation Strategy (Training Split Only)
* **Random Horizontal Flip** ($p=0.5$): Invariant under mirror reflections.
* **Mild Random Rotation** ($\pm 15^\circ$): Simulates handheld capture angles without distorting component geometry.
* **Subtle Color Jitter** (Brightness $\pm 15\%$, Contrast $\pm 15\%$): Adapts to indoor garage and outdoor lighting conditions.
* **Mild Affine Scaling** (Scale $0.95 - 1.05$, Translation $\pm 5\%$): Robustness against distance variations.
* **Deterministic Validation & Test**: Pure deterministic resize ($224 \times 224$) and normalization.

### Class Imbalance Handling
Inverse class frequency weighting is applied directly to the Cross-Entropy loss function:
$$w_c = \frac{N}{K \cdot N_c}$$
Where $N$ is total training samples, $K=10$ is the number of classes, and $N_c$ is the count for class $c$.

---

## 4. CNN Architecture & Design Rationale

### Baseline CNN vs. Optimized SparePartsCNN

```text
[Input: 224 x 224 x 3]
       │
┌──────┴────────────────────────────────────────────────┐
│ 1. EFFICIENT STEM                                     │
│    Conv2D(3 -> 32, 3x3, stride=2, pad=1) -> BN -> ReLU│  [112 x 112]
│    MaxPool2D(2x2, stride=2)                           │  [ 56 x 56]
└──────┬────────────────────────────────────────────────┘
       │
┌──────┴────────────────────────────────────────────────┐
│ 2. STAGE 1 (Low-Level Micro-Texture Extraction)       │
│    Conv2D(32 -> 64, 3x3) -> BN -> ReLU                │
│    ResidualBlock(64 -> 64) with Identity Shortcut     │  [ 56 x 56]
└──────┬────────────────────────────────────────────────┘
       │
┌──────┴────────────────────────────────────────────────┐
│ 3. STAGE 2 (Mid-Level Contour & Part Boundary)        │
│    MaxPool2D(2x2) -> Conv2D(64 -> 128) -> BN -> ReLU  │
│    ResidualBlock(128 -> 128) with Identity Shortcut   │  [ 28 x 28]
└──────┬────────────────────────────────────────────────┘
       │
┌──────┴────────────────────────────────────────────────┐
│ 4. STAGE 3 (High-Level Semantic Assembly)             │
│    MaxPool2D(2x2) -> Conv2D(128 -> 256) -> BN -> ReLU│
│    ResidualBlock(256 -> 256) with Identity Shortcut   │  [ 14 x 14]
└──────┬────────────────────────────────────────────────┘
       │
┌──────┴────────────────────────────────────────────────┐
│ 5. GLOBAL AVERAGE POOLING (GAP)                       │
│    AdaptiveAvgPool2D((1, 1))                          │  [ 1 x 1 x 256]
└──────┬────────────────────────────────────────────────┘
       │
┌──────┴────────────────────────────────────────────────┐
│ 6. REGULARIZED CLASSIFICATION HEAD                    │
│    Dropout(0.4) -> Linear(256 -> 128) -> BatchNorm1d  │
│    ReLU -> Dropout(0.2) -> Linear(128 -> 10 Classes)  │  [ 10 Logits ]
└───────────────────────────────────────────────────────┘
```

### Architectural Comparison (From Scratch)
| Metric | Baseline 3-Stage CNN | Optimized SparePartsCNN | Architectural Impact |
|--------|----------------------|-------------------------|----------------------|
| **Feature Extractor** | Plain Conv Blocks | Stem + 3 Residual Stages | Deeper gradient flow, preserves fine details |
| **Dimensionality Transition** | Flatten ($128 \times 14 \times 14 = 25,088$) | Global Average Pooling ($256 \times 1$) | Translation invariance, eliminates spatial overfitting |
| **Total Parameters** | 6,456,234 (~25.8 MB) | **1,973,290 (~7.52 MB)** | **~70% parameter reduction** |
| **Single-Image Latency** | ~35 ms | **~6.5 ms** | Real-time ready for mobile & web backends |
| **Regularization** | Single dense dropout | Dual-stage dropout + BatchNorm1d | Stronger out-of-distribution generalization |

### Comparative Benchmark: Transfer Learning vs. Custom CNN
To establish a production-grade recognition engine for the automotive marketplace, we conducted an empirical benchmark across industry-standard architectures on the 524-image unseen test set:

| Model Architecture | Test Accuracy (%) | Macro F1 | Precision | Recall | Parameters (M) | Model Size | Role in Project |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Xception** (Selected) 🏆 | **98.09%** | **0.9783** | **0.9784** | **0.9791** | 20.83 M | 79.45 MB | **Primary Marketplace Recognition Engine** |
| **EfficientNet-B0** | 97.33% | 0.9730 | 0.9757 | 0.9719 | 4.02 M | 15.34 MB | High-Efficiency Benchmark |
| **MobileNetV2** | 96.95% | 0.9688 | 0.9710 | 0.9682 | 2.24 M | 8.53 MB | Ultra-Lightweight Mobile Candidate |
| **Inception-V3** | 95.04% | 0.9466 | 0.9470 | 0.9484 | 21.81 M | 83.18 MB | Standard Inception Baseline |
| **Optimized SparePartsCNN** | 77.86% | 0.7741 | 0.7968 | 0.7698 | 1.97 M | 7.50 MB | Custom Scratch Novel Architecture |

> **Selected Production Model**: **Xception** achieved the highest classification accuracy (**98.09%**) and Macro F1 score (**0.9783**), with zero-error performance (100% precision and recall) on Battery, Muffler, and Radiator categories. It serves as the primary recognition engine for live marketplace inference.

---

## 5. Optimization & Training Methodology

* **Optimizer**: AdamW with weight decay ($10^{-4}$) to prevent weight explosion.
* **Learning Rate Schedule**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)`.
* **Early Stopping**: Monitored on validation loss with patience of 6 epochs to prevent overfitting.
* **Model Checkpointing**: Automatically preserves the best weights (`best_spare_parts_model.pth`).
* **Weight Initialization**: Kaiming (He) normal initialization (`nn.init.kaiming_normal_`) matched to ReLU activations.

---

## 6. Directory Structure

```text
d:/s7project/
│
├── dataset/                        # Real labeled spare parts image folders
│   ├── battery/
│   ├── brake_calliper/
│   ├── brake_pad/
│   ├── carburetor/
│   ├── gas_cap/
│   ├── muffler/
│   ├── oil_filter/
│   ├── radiator/
│   ├── spark_plug/
│   └── wheel_rim/
│
├── data/
│   ├── train_val_test_split.json   # Reproducible split manifests (70/15/15)
│   └── marketplace_catalog.json    # E-commerce schema & product catalog
│
├── models/
│   ├── xception.pth                # Primary Xception model weights (98.09% test acc)
│   └── class_mapping.json          # Bidirectional class-to-index mappings
│
├── outputs/
│   ├── accuracy_curve.png          # Training vs Validation accuracy
│   ├── loss_curve.png              # Training vs Validation loss
│   ├── confusion_matrix.png        # Normalized test confusion matrix heatmap
│   ├── sample_predictions.png      # Test prediction grid with confidence
│   ├── correct_predictions.png     # High-confidence correct detections
│   ├── incorrect_predictions.png   # Misclassification error diagnostics
│   ├── evaluation_metrics.json     # Machine-readable test metrics
│   └── classification_report.txt   # Formatted classification report
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration & hyperparameters
│   ├── dataset_inspector.py        # Dataset discovery & integrity scanner
│   ├── data_loader.py              # Augmentation & PyTorch DataLoader pipeline
│   ├── models.py                   # Xception model definition & parameter summary
│   ├── transfer_models.py          # Transfer learning architecture loader
│   ├── train.py                    # Training engine with Early Stopping & LR scheduler
│   ├── evaluate.py                 # Test evaluation & confusion diagnostics
│   ├── visualize.py                # Publication-grade chart generation
│   ├── predict.py                  # Standalone inference API
│   └── marketplace.py              # Marketplace lookup & product card rendering
│
├── main.py                         # Single unified entrypoint
├── requirements.txt                # Dependencies specification
└── README.md                       # Comprehensive documentation
```

---

## 7. Installation & Quickstart

### Prerequisites
* Windows 10/11 or Linux
* Python 3.10+ (tested on Python 3.14)

### Setup Virtual Environment
```powershell
# In project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Full End-to-End Pipeline
```powershell
python main.py
```
This executes the dataset validation, trains the Xception model, computes test evaluation metrics, generates all 6 visualization charts into `outputs/`, and performs a live marketplace prediction demo.

### Run Evaluation & Marketplace Demo (Xception)
To evaluate the **Xception** model on the 524 unseen test images and view live marketplace demonstrations without retraining:
```powershell
python main.py --eval-only
```

### Standalone Image Prediction
Classify any spare part image using the default **Xception** engine with Top-3 probabilities and complete marketplace metadata:
```powershell
python -m src.predict path/to/image.jpg --marketplace
```

Or from Python code:
```python
from src.predict import predict_spare_part

# Uses default Xception (or pass model_name="custom_cnn" / "efficientnet_b0")
result = predict_spare_part("dataset/brake_pad/some_image.jpg")
print(f"Detected: {result['display_name']} ({result['confidence_percent']})")
```

---

## 8. Marketplace Integration Architecture

The prediction engine is designed to seamlessly integrate into an e-commerce marketplace:

```text
[Spare Part Image]
       │
       ▼
[Image Preprocessing (224x224, Normalized)]
       │
       ▼
[OptimizedSparePartsCNN Inference]
       │
       ▼
[Predicted Class: 'brake_pad', Confidence: 96.4%]
       │
       ▼
[MarketplaceCatalog.get_part_info('brake_pad')]
       │
       ├─► Display Name: Low-Dust Ceramic Brake Pad Set
       ├─► Category: Braking System
       ├─► Compatible Vehicles: Toyota RAV4, Honda CR-V, Mazda CX-5, Nissan Rogue
       ├─► Recommended Brands: Akebono, Brembo, Wagner, Bosch
       ├─► Price Range: $34.99 - $95.00 (Avg: $59.99)
       ├─► Availability: In Stock (12 Mo. Warranty)
       └─► Verified Sellers: CeramicBrake Direct (Rating: 4.9), Apex Auto Dynamics (Rating: 4.8)
```
