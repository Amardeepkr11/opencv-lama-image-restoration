import cv2
import numpy as np
import logging

# Configure logger
logger = logging.getLogger("InpaintingTelea")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

def inpaint_telea(image: np.ndarray, mask: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """
    Perform Fast Marching Method (Telea) image inpainting using OpenCV.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array in OpenCV BGR format (uint8).
    mask : np.ndarray
        Binary mask array (uint8). Non-zero pixels represent corrupted/masked areas to fill.
    radius : float, optional
        Inpainting neighborhood radius around pixels to be filled (default: 3.0).
        
    Returns:
    --------
    np.ndarray
        Restored image array in OpenCV BGR format (uint8).
    """
    if image is None or mask is None:
        logger.error("Inpainting failed: Image or mask is None.")
        raise ValueError("Image and mask must not be None.")
        
    # Ensure mask is 2D grayscale uint8
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    # Threshold mask so non-zero pixels are strictly 255 (binary mask)
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        
    logger.info(f"Running OpenCV Telea Inpainting with radius={radius}...")
    
    # Apply OpenCV Telea Inpainting algorithm
    inpainted = cv2.inpaint(image, binary_mask, inpaintRadius=float(radius), flags=cv2.INPAINT_TELEA)
    
    logger.info("OpenCV Telea Inpainting finished successfully.")
    return inpainted
