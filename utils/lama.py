import os
import cv2
import numpy as np
import logging
from PIL import Image
import torch
from simple_lama_inpainting.utils import prepare_img_and_mask, download_model

# Configure logger
logger = logging.getLogger("InpaintingLaMa")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)

def get_active_device() -> torch.device:
    """Detect available compute device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"CUDA available but failed tensor test ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

class SafeSimpleLama:
    """Robust wrapper for SimpleLama model ensuring proper device & map_location handling."""
    def __init__(self, device: torch.device = None) -> None:
        if device is None:
            device = get_active_device()
                
        self.device = device
        logger.info(f"Initializing LaMa Neural Network on device: {self.device}")

        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Lama torchscript model not found: {model_path}")
        else:
            model_path = download_model(LAMA_MODEL_URL)

        # Load TorchScript model with explicit map_location for device safety
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model on {self.device}: {e}. Retrying on CPU...")
            self.device = torch.device("cpu")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        try:
            image_tensor, mask_tensor = prepare_img_and_mask(image, mask, self.device)
            with torch.inference_mode():
                inpainted = self.model(image_tensor, mask_tensor)
                cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(cur_res)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device.type == "cuda":
                logger.error("CUDA Out of Memory encountered! Clearing CUDA cache and switching to CPU fallback...")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model.to(self.device)
                return self.__call__(image, mask)
            else:
                raise e

# Global singleton cache for model instance
_LAMA_MODEL = None

def get_lama_model():
    """Lazy loader for SafeSimpleLama model instance."""
    global _LAMA_MODEL
    if _LAMA_MODEL is None:
        _LAMA_MODEL = SafeSimpleLama()
    return _LAMA_MODEL

def inpaint_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Perform deep-learning based image inpainting using the LaMa (Large Mask Inpainting) model.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array in OpenCV BGR format (uint8).
    mask : np.ndarray
        Binary mask array (uint8). Non-zero pixels represent corrupted/masked areas to fill.
        
    Returns:
    --------
    np.ndarray
        Restored image array in OpenCV BGR format (uint8), matching original dimensions.
    """
    if image is None or mask is None:
        raise ValueError("Image and mask must not be None.")

    orig_h, orig_w = image.shape[:2]

    # 1. Ensure mask is a 2D grayscale uint8 binary mask (0 or 255)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # 2. Convert OpenCV BGR Image and Mask array to PIL Images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(binary_mask).convert("L")

    # 3. Load LaMa model and run inference
    logger.info("Executing LaMa Model inference...")
    lama = get_lama_model()
    result_pil = lama(pil_image, pil_mask)

    # 4. Convert result back to OpenCV BGR numpy array
    result_np = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    # 5. LaMa pads the input up to a multiple of 8 on the bottom/right, so the
    # output must be cropped back; resizing it would shift every pixel.
    if result_bgr.shape[0] >= orig_h and result_bgr.shape[1] >= orig_w:
        result_bgr = result_bgr[:orig_h, :orig_w]
    else:
        result_bgr = cv2.resize(result_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    logger.info("LaMa Model inference completed.")
    return result_bgr
