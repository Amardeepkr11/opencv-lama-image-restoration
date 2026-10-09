import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict, Union

def calculate_unmasked_metrics(original_bgr: np.ndarray, result_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Union[float, str]]:
    """
    Calculate PSNR and SSIM quality metrics ONLY on the non-masked (unaltered) background region.
    
    Parameters:
    -----------
    original_bgr : np.ndarray
        Original source image array (BGR uint8).
    result_bgr : np.ndarray
        Inpainted output image array (BGR uint8).
    mask : np.ndarray
        Binary mask array (255 = inpaint region, 0 = background).
        
    Returns:
    --------
    Dict[str, Union[float, str]]
        Dictionary containing 'psnr' (in dB) and 'ssim' (0.0 to 1.0) scores for the unmasked pixels.
    """
    if original_bgr.shape != result_bgr.shape:
        return {"psnr": 0.0, "ssim": 0.0, "status": "Dimension mismatch"}

    # Identify non-masked pixels (background that should remain identical)
    unmasked_indices = (mask == 0)
    
    # If the entire image was masked, return 0.0
    if not np.any(unmasked_indices):
        return {"psnr": 0.0, "ssim": 0.0, "status": "Entire image masked"}

    # Convert to RGB for SSIM computation
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    res_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # Extract unmasked pixel values for PSNR
    orig_pixels = orig_rgb[unmasked_indices]
    res_pixels = res_rgb[unmasked_indices]

    # Calculate Mean Squared Error (MSE) on unmasked region
    mse = np.mean((orig_pixels.astype(np.float64) - res_pixels.astype(np.float64)) ** 2)
    
    if mse == 0:
        psnr_val = 100.0 # Infinite/Perfect score
    else:
        psnr_val = float(peak_signal_noise_ratio(orig_pixels, res_pixels, data_range=255))

    # Calculate SSIM on full image, masking out inpainting area
    try:
        ssim_val, ssim_full = structural_similarity(
            orig_rgb, res_rgb,
            multichannel=True,
            channel_axis=2 if hasattr(structural_similarity, 'channel_axis') else None,
            full=True,
            data_range=255
        )
        # Average SSIM only over unmasked pixels
        unmasked_ssim = float(np.mean(ssim_full[unmasked_indices]))
    except Exception:
        # Fallback to overall SSIM
        unmasked_ssim = float(structural_similarity(orig_rgb, res_rgb, multichannel=True, data_range=255))

    return {
        "psnr": round(psnr_val, 2),
        "ssim": round(unmasked_ssim, 4),
        "status": "Success"
    }
