import os
import cv2
import numpy as np
from utils.telea import inpaint_telea
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

def add_header(img: np.ndarray, title: str) -> np.ndarray:
    """Add a clean labeled header bar on top of an image panel."""
    h, w, _ = img.shape
    banner_height = 40
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30) # Dark gray header
    
    # Calculate text position
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(title, font, scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_height + text_size[1]) // 2
    
    cv2.putText(banner, title, (text_x, text_y), font, scale, (255, 255, 255), thickness)
    return np.vstack([banner, img])

def main():
    sample_path = os.path.join("samples", "sample_image.png")
    output_dir = "outputs"
    os.makedirs("samples", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load sample image
    if not os.path.exists(sample_path):
        image = create_sample_image(sample_path)
    else:
        image = cv2.imread(sample_path)

    h, w, _ = image.shape

    # 2. Draw rectangular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    rect_top_left = (100, 250)
    rect_bottom_right = (400, 340)
    cv2.rectangle(mask, rect_top_left, rect_bottom_right, 255, -1)

    # 3. Create masked input visualization (red box over corrupted region)
    masked_vis = image.copy()
    masked_vis[mask == 255] = (0, 0, 255)

    # 4. Run Telea algorithm
    print("[+] Running OpenCV Telea Inpainting...")
    telea_res = inpaint_telea(image, mask, radius=3.0)

    # 5. Run LaMa AI model
    print("[+] Running Deep Learning LaMa Inpainting...")
    lama_res = inpaint_lama(image, mask)

    # 6. Build side-by-side comparison panel
    panel1 = add_header(image, "1. Original Image")
    panel2 = add_header(masked_vis, "2. Masked Input")
    panel3 = add_header(telea_res, "3. OpenCV Telea")
    panel4 = add_header(lama_res, "4. AI LaMa Model")

    # Combine side-by-side horizontally
    comparison_grid = np.hstack([panel1, panel2, panel3, panel4])

    # 7. Save output comparison image
    comparison_out = os.path.join(output_dir, "06_comparison.png")
    cv2.imwrite(comparison_out, comparison_grid)

    print("\n[+] Comparison test finished successfully!")
    print(f"    Saved side-by-side comparison to : {comparison_out}")

if __name__ == "__main__":
    main()
