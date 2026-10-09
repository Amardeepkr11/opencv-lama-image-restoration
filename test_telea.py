import os
import cv2
import numpy as np
from utils.telea import inpaint_telea

def create_sample_image(filepath: str) -> np.ndarray:
    """Generate a vibrant synthetic sample image for inpainting tests if none exists."""
    width, height = 500, 500
    # Create smooth color gradient background
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    blue = (xx * 200 + 50).astype(np.uint8)
    green = (yy * 200 + 50).astype(np.uint8)
    red = ((1 - xx) * 150 + 100).astype(np.uint8)
    
    img = cv2.merge([blue, green, red])
    
    # Draw geometric shapes
    cv2.circle(img, (150, 150), 60, (0, 215, 255), -1)   # Gold circle
    cv2.circle(img, (350, 350), 70, (255, 100, 0), -1)   # Blue circle
    cv2.rectangle(img, (300, 100), (420, 220), (50, 205, 50), -1) # Lime green box
    
    # Add text banner
    cv2.putText(img, "CAPSTONE 2026", (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img, "OPENCV TELEA", (130, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, img)
    print(f"[+] Created sample test image at: {filepath}")
    return img

def main():
    # Setup directories
    sample_path = os.path.join("samples", "sample_image.png")
    output_dir = "outputs"
    os.makedirs("samples", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load or create sample image
    if not os.path.exists(sample_path):
        image = create_sample_image(sample_path)
    else:
        image = cv2.imread(sample_path)
        print(f"[+] Loaded existing sample image from: {sample_path}")

    h, w, c = image.shape
    print(f"    Image Resolution: {w}x{h}, Channels: {c}")

    # 2. Draw a rectangular mask (corrupted area to be inpainted)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Rectangular coordinates: (x1, y1) top-left, (x2, y2) bottom-right
    # Target area over part of text and shapes
    rect_top_left = (100, 250)
    rect_bottom_right = (400, 340)
    
    # Fill rectangle with white (255) on the binary mask
    cv2.rectangle(mask, rect_top_left, rect_bottom_right, 255, -1)

    # 3. Create a masked visualization (showing red rectangle over corrupted region on original)
    masked_visualization = image.copy()
    masked_visualization[mask == 255] = (0, 0, 255) # Paint masked region bright red for visual clarity

    # 4. Perform OpenCV Telea Inpainting
    print("[+] Running Telea Inpainting (radius=3)...")
    inpainted_result = inpaint_telea(image, mask, radius=3.0)

    # 5. Save outputs into outputs/ folder
    original_out = os.path.join(output_dir, "01_original.png")
    mask_out = os.path.join(output_dir, "02_mask.png")
    masked_out = os.path.join(output_dir, "03_masked_input.png")
    result_out = os.path.join(output_dir, "04_telea_result.png")

    cv2.imwrite(original_out, image)
    cv2.imwrite(mask_out, mask)
    cv2.imwrite(masked_out, masked_visualization)
    cv2.imwrite(result_out, inpainted_result)

    print("\n[+] Telea Inpainting test finished successfully!")
    print(f"    Saved original image to : {original_out}")
    print(f"    Saved binary mask to     : {mask_out}")
    print(f"    Saved masked preview to  : {masked_out}")
    print(f"    Saved Telea result to    : {result_out}")

if __name__ == "__main__":
    main()
