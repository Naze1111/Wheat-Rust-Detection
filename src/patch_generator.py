import cv2
import os
from pathlib import Path

def generate_patches(input_dir, output_dir, patch_size=300, overlap=0.3, sky_threshold=200):
    """
    Splits images into overlapping patches.
    - Large images (e.g. wide field shots): sliced into multiple patches.
    - Small images (close-ups smaller than patch_size): copied through as-is,
      resized up to patch_size so they still work with the classifier.
    - Discards patches that are mostly sky (bright, low-texture background).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = int(patch_size * (1 - overlap))

    # Accept multiple extensions, not just .jpg
    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.jfif", "*.JFIF"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.glob(ext))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {img_path.name}, skipping")
            continue

        h, w = img.shape[:2]
        base_name = img_path.stem

        # Case 1: image smaller than patch_size in either dimension
        # -> treat as a close-up, resize up, save as a single "patch"
        if h < patch_size or w < patch_size:
            resized = cv2.resize(img, (patch_size, patch_size))
            out_path = output_dir / f"{base_name}_full.jpg"
            cv2.imwrite(str(out_path), resized)
            print(f"{img_path.name}: too small to slice, resized and saved as-is")
            continue

        # Case 2: image big enough to slide a window across
        patch_idx = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img[y:y+patch_size, x:x+patch_size]

                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                if gray.mean() > sky_threshold and gray.std() < 15:
                    continue  # skip mostly-sky patches

                out_path = output_dir / f"{base_name}_p{patch_idx}.jpg"
                cv2.imwrite(str(out_path), patch)
                patch_idx += 1

        print(f"{img_path.name}: generated {patch_idx} patches")

if __name__ == "__main__":
    generate_patches("data/field_images/healthy", "data/patches/healthy")
    generate_patches("data/field_images/rust", "data/patches/rust")