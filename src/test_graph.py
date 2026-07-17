import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = 300
PATCH_SIZE = 300
STRIDE = 210
TEST_DIR = Path("data/test_images/prof_dataset")
OUTPUT_DIR = Path("outputs/test_results_3")
GRAPH_DIR = Path("outputs/test_results_3/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model("models/efficientnet_rust_classifier.h5")

extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.jfif", "*.JFIF"]
image_paths = []
for ext in extensions:
    image_paths.extend(TEST_DIR.glob(ext))

summary_records = []  # collects one row per image for the final summary chart

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: could not read {img_path.name}, skipping")
        continue

    h, w = img.shape[:2]
    patch_preds = []

    if h < PATCH_SIZE or w < PATCH_SIZE:
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
        patch_preds.append(pred)
    else:
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE)) / 255.0
                pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
                patch_preds.append(pred)

    patch_preds = np.array(patch_preds)
    pct_rust_patches = 100 * np.mean(patch_preds > 0.5)
    avg_confidence = patch_preds.mean()
    min_confidence = patch_preds.min()
    max_confidence = patch_preds.max()
    label = "RUST" if pct_rust_patches > 20 else "HEALTHY"
    color = (0, 0, 255) if label == "RUST" else (0, 200, 0)

    # --- Annotate and save the image (as before, now with range added) ---
    annotated = img.copy()
    text = f"{label} ({pct_rust_patches:.0f}% patches, avg {avg_confidence:.2f}, range {min_confidence:.2f}-{max_confidence:.2f})"
    font_scale = max(1.0, img.shape[1] / 800)
    thickness = max(2, int(font_scale * 2))
    cv2.putText(annotated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

    out_path = OUTPUT_DIR / f"{label.lower()}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

    # --- NEW: per-image histogram of patch scores ---
    plt.figure(figsize=(6, 4))
    plt.hist(patch_preds, bins=20, range=(0, 1), color="firebrick" if label == "RUST" else "seagreen", edgecolor="black")
    plt.axvline(0.5, color="black", linestyle="--", label="decision threshold (0.5)")
    plt.title(f"Patch score distribution: {img_path.name}")
    plt.xlabel("Model confidence (0 = healthy, 1 = rust)")
    plt.ylabel("Number of patches")
    plt.legend()
    plt.tight_layout()
    hist_path = GRAPH_DIR / f"hist_{img_path.stem}.png"
    plt.savefig(hist_path)
    plt.close()

    summary_records.append({
        "filename": img_path.name,
        "label": label,
        "pct_rust_patches": pct_rust_patches,
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence
    })

    print(f"{img_path.name}: {label} — {pct_rust_patches:.0f}% of {len(patch_preds)} patches flagged "
          f"(avg {avg_confidence:.2f}, range {min_confidence:.2f}-{max_confidence:.2f}) -> {out_path}")

# --- NEW: summary bar chart across all test images ---
if summary_records:
    names = [r["filename"] for r in summary_records]
    pcts = [r["pct_rust_patches"] for r in summary_records]
    colors = ["firebrick" if r["label"] == "RUST" else "seagreen" for r in summary_records]

    plt.figure(figsize=(max(8, len(names) * 0.5), 5))
    plt.bar(range(len(names)), pcts, color=colors)
    plt.axhline(20, color="black", linestyle="--", label="classification threshold (20%)")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("% of patches flagged as rust")
    plt.title("Rust severity across all test images")
    plt.legend()
    plt.tight_layout()
    summary_path = GRAPH_DIR / "summary_all_images.png"
    plt.savefig(summary_path)
    plt.close()
    print(f"\nSummary chart saved to: {summary_path.resolve()}")

print(f"\nAll annotated images saved to: {OUTPUT_DIR.resolve()}")
print(f"All graphs saved to: {GRAPH_DIR.resolve()}")