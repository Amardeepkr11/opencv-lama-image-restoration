import os
import cv2
import numpy as np
from utils.lama import inpaint_lama

def create_sample_image(filepath: str) -> np.ndarray:
    """Generate sample image if missing (same logic as test_telea.py)."""
    width, height = 500, 500
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    blue = (xx * 200 + 50).astype(np.uint8)
    green = (yy * 200 + 50).astype(np.uint8)
    red = ((1 - xx) * 150 + 100).astype(np.uint8)
    
    img = cv2.merge([blue, green, red])
    cv2.circle(img, (150, 150), 60, (0, 215, 255), -1)
    cv2.circle(img, (350, 350), 70, (255, 100, 0), -1)
    cv2.rectangle(img, (300, 100), (420, 220), (50, 205, 50), -1)
    
    cv2.putText(img, "CAPSTONE 2026", (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img, "OPENCV TELEA", (130, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, img)
    return img

def main():
    sample_path = os.path.join("samples", "sample_image.png")
    output_dir = "outputs"
    os.makedirs("samples", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load or generate identical sample image
    if not os.path.exists(sample_path):
        image = create_sample_image(sample_path)
    else:
        image = cv2.imread(sample_path)
        print(f"[+] Loaded sample image from: {sample_path}")

    h, w, _ = image.shape

    # 2. Draw identical rectangular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    rect_top_left = (100, 250)
    rect_bottom_right = (400, 340)
    cv2.rectangle(mask, rect_top_left, rect_bottom_right, 255, -1)

    # 3. Perform LaMa Inpainting
    print("[+] Running AI LaMa Deep Learning Inpainting...")
    lama_result = inpaint_lama(image, mask)

    # 4. Save result into outputs/ folder
    result_out = os.path.join(output_dir, "05_lama_result.png")
    cv2.imwrite(result_out, lama_result)

    print("\n[+] LaMa Inpainting test finished successfully!")
    print(f"    Saved LaMa result to : {result_out}")

if __name__ == "__main__":
    main()
