"""
Generate comprehensive, publication-grade comparative benchmark charts
for all evaluated models on the Car Spare Parts dataset (524 unseen test images).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output destination
OUTPUT_FILE = Path("outputs/model_comparison.png")

# Benchmark Data
models = [
    "Xception\n(Selected / Best)",
    "EfficientNet-B0",
    "MobileNetV2",
    "Inception-V3",
    "Optimized CNN\n(Custom)",
    "Baseline CNN\n(3-Stage)"
]

accuracy = [98.09, 97.33, 96.95, 95.04, 77.86, 64.12]
f1_score = [0.9783, 0.9730, 0.9688, 0.9466, 0.7741, 0.6380]
precision = [0.9784, 0.9757, 0.9710, 0.9470, 0.7968, 0.6510]
recall = [0.9791, 0.9719, 0.9682, 0.9484, 0.7698, 0.6350]
parameters_m = [20.83, 4.02, 2.24, 21.81, 1.97, 6.46]
model_size_mb = [79.45, 15.34, 8.53, 83.18, 7.50, 25.80]

colors = [
    "#2563eb",  # Vibrant Blue for Winner (Xception)
    "#059669",  # Emerald Green
    "#0891b2",  # Cyan
    "#7c3aed",  # Purple
    "#d97706",  # Amber
    "#dc2626",  # Red
]

fig = plt.figure(figsize=(16, 11), dpi=200)
fig.patch.set_facecolor("#ffffff")

# Grid layout: 2 rows x 2 columns
# 1. Test Accuracy & F1 Comparison
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(models))
width = 0.38

rects1 = ax1.bar(x - width/2, accuracy, width, label="Test Accuracy (%)", color="#1e40af", edgecolor="black", linewidth=0.8, alpha=0.9)
rects2 = ax1.bar(x + width/2, [f * 100 for f in f1_score], width, label="Macro F1-Score (%)", color="#10b981", edgecolor="black", linewidth=0.8, alpha=0.9)

ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax1.set_title("Test Accuracy & Macro F1-Score Comparison", fontsize=14, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9.5, fontweight="semibold")
ax1.set_ylim(50, 108)
ax1.legend(loc="lower right", frameon=True, fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.35)

# Add values above bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#1e40af")
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#047857")

# Highlight winner
rects1[0].set_color("#2563eb")
rects1[0].set_edgecolor("#fbbf24")
rects1[0].set_linewidth(2.0)

# 2. Precision vs Recall
ax2 = plt.subplot(2, 2, 2)
rects_p = ax2.bar(x - width/2, [p * 100 for p in precision], width, label="Precision (Macro %)", color="#6366f1", edgecolor="black", linewidth=0.8, alpha=0.85)
rects_r = ax2.bar(x + width/2, [r * 100 for r in recall], width, label="Recall (Macro %)", color="#f59e0b", edgecolor="black", linewidth=0.8, alpha=0.85)

ax2.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax2.set_title("Precision vs. Recall Comparison", fontsize=14, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9.5, fontweight="semibold")
ax2.set_ylim(50, 108)
ax2.legend(loc="lower right", frameon=True, fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.35)

for rect in rects_p:
    h = rect.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#4338ca")
for rect in rects_r:
    h = rect.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#b45309")

# 3. Model Size & Footprint
ax3 = plt.subplot(2, 2, 3)
bars_size = ax3.bar(models, model_size_mb, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
ax3.set_ylabel("Model Size on Disk (MB)", fontsize=12, fontweight="bold")
ax3.set_title("Model Footprint & Storage Efficiency", fontsize=14, fontweight="bold", pad=12)
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, fontsize=9.5, fontweight="bold")
ax3.set_ylim(0, 100)
ax3.grid(axis="y", linestyle="--", alpha=0.35)

for bar, param in zip(bars_size, parameters_m):
    h = bar.get_height()
    ax3.annotate(f"{h:.1f} MB\n({param:.1f}M params)", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

# 4. Accuracy vs Parameter Efficiency (Trade-off Landscape)
ax4 = plt.subplot(2, 2, 4)
custom_offsets = [
    (-3.6, 2.0),   # Xception
    (0.8, 1.8),    # EfficientNet-B0
    (-2.2, -3.8),  # MobileNetV2
    (1.0, -1.8),   # Inception-V3
    (0.8, 1.2),    # Optimized CNN
    (0.8, 1.2)     # Baseline CNN
]

for i, name in enumerate(models):
    ax4.scatter(parameters_m[i], accuracy[i], s=model_size_mb[i] * 12 + 100, color=colors[i], alpha=0.75, edgecolors="black", linewidth=1.5, zorder=5)
    ox, oy = custom_offsets[i]
    clean_name = name.split("\n")[0]
    ax4.annotate(f"{clean_name}\n({accuracy[i]:.1f}%)", xy=(parameters_m[i], accuracy[i]),
                 xytext=(parameters_m[i] + ox, accuracy[i] + oy),
                 fontsize=9.5, fontweight="bold", color="#1f2937")

ax4.set_xlabel("Total Parameters (Millions)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
ax4.set_title("Accuracy vs. Parameter Complexity Trade-off", fontsize=14, fontweight="bold", pad=12)
ax4.set_xlim(-2, 26)
ax4.set_ylim(60, 105)
ax4.grid(True, linestyle="--", alpha=0.4)

# Benchmark note
ax4.axhline(y=98.09, color="#2563eb", linestyle=":", alpha=0.6, label="Xception Top Mark (98.09%)")
ax4.legend(loc="lower right", fontsize=9.5, frameon=True)

plt.suptitle("Comparative Evaluation Across All Trained Models (524 Unseen Test Images)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()
print(f"[SUCCESS] Saved model comparison chart to {OUTPUT_FILE}")
