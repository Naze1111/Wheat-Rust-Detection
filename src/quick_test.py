import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 300
PATCH_SIZE = 300
STRIDE = 210  # ~30% overlap, matches training patch_generator settings
TEST_DIR = Path("data/test_images/test_1_effnet")
OUTPUT_DIR = Path("outputs/test_results_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model("models/efficientnet_rust_classifier.h5")

extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.jfif", "*.JFIF"]
image_paths = []
for ext in extensions:
    image_paths.extend(TEST_DIR.glob(ext))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: could not read {img_path.name}, skipping")
        continue

    h, w = img.shape[:2]
    patch_preds = []

    if h < PATCH_SIZE or w < PATCH_SIZE:
        # Small image (close-up): predict on the whole thing, resized
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
        patch_preds.append(pred)
    else:
        # Large image: slide across it just like training did
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE)) / 255.0
                pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
                patch_preds.append(pred)

    patch_preds = np.array(patch_preds)
    pct_rust_patches = 100 * np.mean(patch_preds > 0.5)
    avg_confidence = patch_preds.mean()
    label = "RUST" if pct_rust_patches > 20 else "HEALTHY"  # tune this threshold
    color = (0, 0, 255) if label == "RUST" else (0, 200, 0)

    annotated = img.copy()
    text = f"{label} ({pct_rust_patches:.0f}% patches flagged, avg {avg_confidence:.2f})"
    font_scale = max(1.0, img.shape[1] / 800)
    thickness = max(2, int(font_scale * 2))
    cv2.putText(annotated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

    out_path = OUTPUT_DIR / f"{label.lower()}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

    print(f"{img_path.name}: {label} — {pct_rust_patches:.0f}% of {len(patch_preds)} patches flagged rust (avg conf {avg_confidence:.2f}) -> {out_path}")