import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 300
TEST_DIR = Path("data/test_images")
OUTPUT_DIR = Path("outputs/test_results")
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

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
    label = "RUST" if pred > 0.5 else "HEALTHY"
    color = (0, 0, 255) if label == "RUST" else (0, 200, 0)  # red for rust, green for healthy

    # Draw label + confidence directly onto a copy of the original image
    annotated = img.copy()
    text = f"{label} ({pred:.2f})"
    font_scale = max(1.0, img.shape[1] / 800)  # scale text size to image width
    thickness = max(2, int(font_scale * 2))
    cv2.putText(annotated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

    out_path = OUTPUT_DIR / f"{label.lower()}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

    print(f"{img_path.name}: {label} (confidence {pred:.2f}) -> saved to {out_path}")

print(f"\nAll annotated images saved to: {OUTPUT_DIR.resolve()}")