import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple

def resize_image_aspect(image: Union[np.ndarray, Image.Image], max_dim: int = 1024) -> Tuple[Union[np.ndarray, Image.Image], Tuple[int, int]]:
    """
    Resize an image preserving its exact aspect ratio if the longer dimension exceeds max_dim.
    
    Parameters:
    -----------
    image : Union[np.ndarray, Image.Image]
        Input image as NumPy BGR array or PIL Image.
    max_dim : int
        Maximum allowed dimension for the longer side (width or height).
        
    Returns:
    --------
    Tuple[Union[np.ndarray, Image.Image], Tuple[int, int]]
        Resized image (same type as input) and original dimensions (orig_w, orig_h).
    """
    if isinstance(image, Image.Image):
        orig_w, orig_h = image.size
        if max(orig_w, orig_h) > max_dim:
            if orig_w > orig_h:
                new_w = max_dim
                new_h = int(orig_h * (max_dim / orig_w))
            else:
                new_h = max_dim
                new_w = int(orig_w * (max_dim / orig_h))
            resized_img = image.resize((new_w, new_h), Image.LANCZOS)
            return resized_img, (orig_w, orig_h)
        return image, (orig_w, orig_h)
    
    elif isinstance(image, np.ndarray):
        orig_h, orig_w = image.shape[:2]
        if max(orig_w, orig_h) > max_dim:
            if orig_w > orig_h:
                new_w = max_dim
                new_h = int(orig_h * (max_dim / orig_w))
            else:
                new_h = max_dim
                new_w = int(orig_w * (max_dim / orig_h))
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized_img, (orig_w, orig_h)
        return image, (orig_w, orig_h)
    
    else:
        raise ValueError("Unsupported image type. Expected PIL Image or NumPy array.")

def convert_cv_to_bytes(cv_img_bgr: np.ndarray, ext: str = ".png") -> bytes:
    """
    Convert an OpenCV BGR NumPy image array into byte buffer format for downloading.
    
    Parameters:
    -----------
    cv_img_bgr : np.ndarray
        Input image in OpenCV BGR format.
    ext : str
        Target image format extension (e.g. '.png', '.jpg').
        
    Returns:
    --------
    bytes
        Encoded image byte buffer.
    """
    if cv_img_bgr is None or cv_img_bgr.size == 0:
        return b""
    is_success, buffer = cv2.imencode(ext, cv_img_bgr)
    if is_success:
        return buffer.tobytes()
    return b""
