import cv2
import numpy as np
from pathlib import Path

def segment_rust_regions(img_path, output_path=None):
    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_rust = np.array([5, 50, 50])
    upper_rust = np.array([25, 255, 255])
    rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)

    kernel = np.ones((5, 5), np.uint8)
    rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
    rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)

    affected_pct = 100 * np.count_nonzero(rust_mask) / rust_mask.size

    if output_path:
        overlay = img.copy()
        overlay[rust_mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(output_path), blended)

    return affected_pct, rust_mask

if __name__ == "__main__":
    test_dir = Path("data/test_images")
    out_dir = Path("outputs/color_segmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_dir.glob("*.[jJ][pP][gG]"):
        pct, mask = segment_rust_regions(img_path, out_dir / f"seg_{img_path.name}")
        print(f"{img_path.name}: ~{pct:.1f}% of pixels flagged as possible rust")