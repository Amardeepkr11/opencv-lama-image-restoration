import cv2
import numpy as np
from typing import Tuple, Callable, Optional, List

def extract_binary_mask(canvas_data: Optional[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Extract binary mask from canvas RGBA image data and resize to target_shape.
    
    Parameters:
    -----------
    canvas_data : Optional[np.ndarray]
        Canvas RGBA array from streamlit-drawable-canvas.
    target_shape : Tuple[int, int]
        Desired (height, width) matching the processing image.
        
    Returns:
    --------
    np.ndarray
        Single-channel uint8 binary mask array (255 = masked region, 0 = background).
    """
    target_h, target_w = target_shape
    if canvas_data is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    canvas_h, canvas_w = canvas_data.shape[:2]
    
    # Extract alpha channel and RGB painted stroke signals
    alpha_channel = canvas_data[:, :, 3]
    rgb_sum = canvas_data[:, :, :3].sum(axis=2)
    
    raw_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    raw_mask[(alpha_channel > 0) | (rgb_sum > 0)] = 255

    # Resize mask to processing target_shape using NEAREST NEIGHBOR to preserve crisp pixel alignment
    if (canvas_h, canvas_w) != (target_h, target_w):
        binary_mask = cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        binary_mask = raw_mask

    # Re-threshold to strictly binary 0 or 255
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def build_mask_from_strokes(strokes: List[dict], target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Render click-based mask strokes into a binary mask of size target_shape.

    Each stroke stores coordinates normalised to 0..1 of the image size, so the
    mask is independent of the size the image was displayed at:
      {"tool": "dot",   "erase": bool, "x": float, "y": float, "r": float}
      {"tool": "rect",  "erase": bool, "x1": float, "y1": float, "x2": float, "y2": float}
      {"tool": "line",  "erase": bool, "points": [[x, y], ...], "r": float}
    "r" is normalised to the image width. Strokes are applied in order, so an
    erase stroke removes whatever was painted before it.
    Freehand "line" points use 0..1 as the outer edges of the image (browser
    convention), so pixel i covers [i/w, (i+1)/w).
    """
    target_h, target_w = target_shape
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    shift = 4
    scale = 1 << shift

    for s in strokes:
        value = 0 if s.get("erase") else 255
        if s["tool"] == "line":
            radius_px = max(0.5, s["r"] * target_w)
            pts = [(int(round((x * target_w - 0.5) * scale)), int(round((y * target_h - 0.5) * scale)))
                   for x, y in s["points"]]
            for p in pts:
                cv2.circle(mask, p, int(round(radius_px * scale)), value, thickness=-1, shift=shift)
            thickness = max(1, int(round(2 * radius_px)))
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(mask, p, q, value, thickness=thickness, shift=shift)
        elif s["tool"] == "rect":
            x1, x2 = sorted((s["x1"], s["x2"]))
            y1, y2 = sorted((s["y1"], s["y2"]))
            p1 = (int(round(x1 * (target_w - 1))), int(round(y1 * (target_h - 1))))
            p2 = (int(round(x2 * (target_w - 1))), int(round(y2 * (target_h - 1))))
            cv2.rectangle(mask, p1, p2, value, thickness=-1)
        else:
            center = (int(round(s["x"] * (target_w - 1))), int(round(s["y"] * (target_h - 1))))
            radius = max(1, int(round(s["r"] * target_w)))
            cv2.circle(mask, center, radius, value, thickness=-1)

    return mask

def dilate_mask(mask: np.ndarray, dilation_radius: int = 3) -> np.ndarray:
    """
    Slightly expand/dilate mask boundaries to prevent edge artifacts or halos.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask (uint8).
    dilation_radius : int
        Radius in pixels to dilate mask (0 = no dilation).
        
    Returns:
    --------
    np.ndarray
        Dilated binary mask.
    """
    if dilation_radius <= 0 or np.count_nonzero(mask) == 0:
        return mask.copy()
        
    kernel_size = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated

def create_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Generate a visual preview of the original image with a semi-transparent red mask overlay.
    
    Parameters:
    -----------
    image_bgr : np.ndarray
        Original image in OpenCV BGR format.
    mask : np.ndarray
        Binary mask array.
    alpha : float
        Opacity of the red mask overlay (0.0 to 1.0).
        
    Returns:
    --------
    np.ndarray
        Image array with semi-transparent red mask preview overlay.
    """
    overlay = image_bgr.copy()
    red_color = np.array([0, 0, 255], dtype=np.uint8) # BGR Red
    
    mask_indices = mask > 0
    if np.any(mask_indices):
        overlay[mask_indices] = cv2.addWeighted(
            image_bgr[mask_indices], 1.0 - alpha,
            np.tile(red_color, (np.sum(mask_indices), 1)), alpha, 0
        )
    return overlay

def get_mask_bounding_box(mask: np.ndarray, padding: int = 40) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box coordinates around non-zero mask region with generous padding.
    """
    h, w = mask.shape[:2]
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0, h, 0, w
        
    ymin = max(0, int(np.min(y_indices)) - padding)
    ymax = min(h, int(np.max(y_indices)) + padding + 1)
    xmin = max(0, int(np.min(x_indices)) - padding)
    xmax = min(w, int(np.max(x_indices)) + padding + 1)
    
    return ymin, ymax, xmin, xmax

def smart_crop_inpaint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    inpaint_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    padding: int = 50,
    blend_feather: int = 15,
    max_coverage_threshold: float = 0.45
) -> np.ndarray:
    """
    Accelerate inpainting by cropping a localized region around the mask, 
    inpainting the crop, and soft-blending it back into the full image.
    Falls back to full image inpainting if mask covers a large portion of the image.
    """
    h, w = image_bgr.shape[:2]
    total_pixels = h * w
    masked_pixels = np.count_nonzero(mask)
    
    if masked_pixels == 0:
        return image_bgr.copy()
        
    coverage = masked_pixels / float(total_pixels)
    
    # If mask coverage is large, perform full image inpainting directly
    if coverage >= max_coverage_threshold:
        return inpaint_fn(image_bgr, mask)
        
    # Get padded bounding box
    ymin, ymax, xmin, xmax = get_mask_bounding_box(mask, padding=padding)
    
    crop_img = image_bgr[ymin:ymax, xmin:xmax].copy()
    crop_mask = mask[ymin:ymax, xmin:xmax].copy()
    
    # Run inpainting model/algorithm on cropped region
    inpainted_crop = inpaint_fn(crop_img, crop_mask)
    
    # Soft alpha-feathered blending along boundary edges
    result = image_bgr.copy()
    crop_h, crop_w = crop_img.shape[:2]
    
    # Create smooth blend mask
    blend_mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    blend_mask[crop_mask > 0] = 1.0
    
    if blend_feather > 0:
        ksize = 2 * blend_feather + 1
        blend_mask = cv2.GaussianBlur(blend_mask, (ksize, ksize), 0)
        
    blend_mask_3d = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)
    
    # Blend crop into full image
    blended_region = (inpainted_crop.astype(np.float32) * blend_mask_3d + 
                      crop_img.astype(np.float32) * (1.0 - blend_mask_3d))
                      
    result[ymin:ymax, xmin:xmax] = np.clip(blended_region, 0, 255).astype(np.uint8)
    return result

def get_zoomed_crop(image_bgr: np.ndarray, mask: np.ndarray, crop_size: int = 220) -> np.ndarray:
    """
    Extract a square crop centered around the mask centroid for close-up visual comparison.
    """
    h, w = image_bgr.shape[:2]
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(np.mean(y_indices)), int(np.mean(x_indices))
        
    half_size = crop_size // 2
    ymin = max(0, cy - half_size)
    ymax = min(h, cy + half_size)
    xmin = max(0, cx - half_size)
    xmax = min(w, cx + half_size)
    
    return image_bgr[ymin:ymax, xmin:xmax]
