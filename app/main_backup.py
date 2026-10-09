import os
import time
import logging
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Import modular utilities
from utils.brush_canvas import brush_canvas
from utils.image_coordinates import image_coordinates
from utils.image_io import resize_image_aspect, convert_cv_to_bytes
from utils.mask import (
    build_mask_from_strokes,
    dilate_mask,
    create_mask_overlay,
    smart_crop_inpaint,
    get_zoomed_crop,
    get_mask_bounding_box,
)
from utils.metrics import calculate_unmasked_metrics
from utils.telea import inpaint_telea
from utils.lama import inpaint_lama, get_lama_model, get_active_device

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("AppMain")

# Page Configuration
st.set_page_config(
    page_title="AI-Based Image Inpainting System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit Model Cache
@st.cache_resource
def load_cached_lama_model():
    """Cache LaMa neural network model instance in memory across sessions."""
    logger.info("Loading cached LaMa model instance...")
    return get_lama_model()

def safe_rerun():
    """Version-safe rerun helper for Streamlit 1.23.1."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def main():
    st.title("🎨 AI-Based Image Inpainting System")
    st.caption("B.Tech Capstone Project: Object & Text Removal using OpenCV Telea & Deep Learning (LaMa)")

    # ---------------------------------------------------------
    # SIDEBAR CONFIGURATION & SETTINGS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ System Configuration")

    # Inpainting Method Selection
    method = st.sidebar.radio(
        "Select Inpainting Method:",
        ["OpenCV Telea", "LaMa AI Model", "Both (Compare Side-by-Side)"],
        index=2
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Masking & Brush Tools")

    tool_choice = st.sidebar.selectbox(
        "Drawing Tool",
        ["Freehand Brush 🖌️ (drag)", "Freehand Eraser 🧽 (drag)",
         "Brush Dot 🔴 (click)", "Eraser Dot 🧹 (click)", "Rectangle Box 🔲 (click & drag)"],
        index=0
    )
    is_freehand = "Freehand" in tool_choice
    is_rect_tool = "Rectangle" in tool_choice
    is_eraser = "Eraser" in tool_choice

    brush_size = st.sidebar.slider("Brush Size (px)", min_value=5, max_value=120, value=25)
    mask_dilation = st.sidebar.slider("Mask Dilation / Expansion (px)", min_value=0, max_value=15, value=3,
                                      help="Slightly expands mask boundaries to prevent edge halos or residual outlines.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Performance & Model Settings")

    # Algorithm specific parameters
    telea_radius = 3.0
    if "Telea" in method or "Both" in method:
        telea_radius = float(st.sidebar.slider("Telea Radius", min_value=1.0, max_value=15.0, value=3.0))

    max_proc_size = st.sidebar.select_slider(
        "Max Processing Dimension (px)",
        options=[512, 768, 1024, 1280, 1536, 2048],
        value=1024,
        help="Limits max image side during inpainting processing to prevent memory issues."
    )

    use_smart_crop = st.sidebar.checkbox(
        "Enable Smart Bounding-Box Crop",
        value=True,
        help="Crops region around mask for up to 5x faster LaMa inference, then soft-blends back into full image."
    )

    # Active Hardware Device Info Badge
    active_device = get_active_device()
    device_label = "🟢 GPU (CUDA Accelerated)" if active_device.type == "cuda" else "💻 CPU Mode"
    st.sidebar.markdown(f"**Compute Hardware:** `{device_label}`")

    # Session State for mask strokes (stored in 0..1 image coordinates)
    if "canvas_key_id" not in st.session_state:
        st.session_state.canvas_key_id = 0
    if "mask_strokes" not in st.session_state:
        st.session_state.mask_strokes = []
    if "last_click_time" not in st.session_state:
        st.session_state.last_click_time = None
    if "last_stroke_id" not in st.session_state:
        st.session_state.last_stroke_id = 0

    # Callbacks instead of st.rerun(): stopping the run here, before the file
    # uploader below is reached, makes Streamlit 1.23 drop the uploaded image.
    def undo_last_stroke():
        if st.session_state.mask_strokes:
            st.session_state.mask_strokes.pop()

    def clear_mask():
        st.session_state.mask_strokes = []
        st.session_state.canvas_key_id += 1

    undo_col, clear_col = st.sidebar.columns(2)
    undo_col.button("↩️ Undo", use_container_width=True, on_click=undo_last_stroke)
    clear_col.button("🧹 Clear Mask", use_container_width=True, on_click=clear_mask)

    # ---------------------------------------------------------
    # STEP 1: IMAGE INPUT & SELECTION
    # ---------------------------------------------------------
    st.header("Step 1: Upload Image")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Image File Source")
    
    uploaded_file = st.sidebar.file_uploader("Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    use_sample = st.sidebar.checkbox("Use Demo Sample Image", value=uploaded_file is None)

    pil_image_raw = None
    if uploaded_file is not None:
        try:
            pil_image_raw = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Failed to load uploaded image: {e}")
            return
    elif use_sample:
        if os.path.exists("samples/large_test.jpg"):
            sample_path = "samples/large_test.jpg"
        else:
            sample_path = os.path.join("samples", "sample_image.png")
            if not os.path.exists(sample_path):
                from test_telea import create_sample_image
                create_sample_image(sample_path)
        try:
            pil_image_raw = Image.open(sample_path).convert("RGB")
        except Exception as e:
            st.error(f"❌ Error loading sample image: {e}")
            return

    if pil_image_raw is None:
        st.info("👈 Please upload an image or select 'Use Demo Sample Image' in the sidebar.")
        return

    # Resize raw image to processing size (max_proc_size) preserving aspect ratio
    proc_pil, (orig_w, orig_h) = resize_image_aspect(pil_image_raw, max_dim=max_proc_size)
    proc_w, proc_h = proc_pil.size
    proc_bgr = cv2.cvtColor(np.array(proc_pil), cv2.COLOR_RGB2BGR)

    # Resize for UI Canvas display (max display width ~700px)
    disp_max_dim = 700
    disp_pil, _ = resize_image_aspect(proc_pil, max_dim=disp_max_dim)
    disp_w, disp_h = int(disp_pil.size[0]), int(disp_pil.size[1])

    # A new image invalidates the strokes painted on the previous one
    image_signature = (uploaded_file.name if uploaded_file is not None else "sample", orig_w, orig_h)
    if st.session_state.get("image_signature") != image_signature:
        st.session_state.image_signature = image_signature
        st.session_state.mask_strokes = []
        st.session_state.last_click_time = None
        st.session_state.canvas_key_id += 1

    # Informative resolution banner
    st.info(
        f"📏 **Image Info**: Original `{orig_w}x{orig_h}px` | "
        f"Processing Resolution `{proc_w}x{proc_h}px` | "
        f"Canvas Display `{disp_w}x{disp_h}px`"
    )

    # ---------------------------------------------------------
    # STEP 2: INTERACTIVE MASK DRAWING & PREVIEW
    # ---------------------------------------------------------
    st.header("Step 2: Paint Mask & Preview Region")

    st.subheader("Interactive Mask Painter")
    if is_freehand and is_eraser:
        st.caption("Hold the mouse button and drag to erase parts of the red mask.")
    elif is_freehand:
        st.caption("Hold the mouse button and drag to paint red over the object, text, or region you want to remove.")
    elif is_rect_tool:
        st.caption("Press, drag and release over the region you want to remove.")
    else:
        st.caption("Click on the image to paint red dots over the object, text, or region you want to remove.")

    # Show the display image with the current mask painted on top of it
    disp_mask = build_mask_from_strokes(st.session_state.mask_strokes, target_shape=(disp_h, disp_w))
    disp_rgb = np.array(disp_pil)
    disp_overlay_rgb = cv2.cvtColor(
        create_mask_overlay(cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR), disp_mask, alpha=0.55),
        cv2.COLOR_BGR2RGB
    )

    if is_freehand:
        result = brush_canvas(
            Image.fromarray(disp_overlay_rgb),
            width=disp_w,
            brush_radius=(brush_size / 2.0) / float(disp_w),
            erase=is_eraser,
            ack_id=st.session_state.last_stroke_id,
            key=f"brush_{st.session_state.canvas_key_id}",
        )
        new_strokes = sorted(
            (s for s in (result or {}).get("strokes", []) if s.get("id", 0) > st.session_state.last_stroke_id),
            key=lambda s: s["id"],
        )
        if new_strokes:
            for s in new_strokes:
                st.session_state.mask_strokes.append({
                    "tool": "line", "erase": bool(s.get("erase")), "r": float(s["r"]),
                    "points": [[min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)] for x, y in s["points"]],
                })
            st.session_state.last_stroke_id = int(new_strokes[-1]["id"])
            safe_rerun()
        click = None
    else:
        click = image_coordinates(
            Image.fromarray(disp_overlay_rgb),
            width=disp_w,
            click_and_drag=is_rect_tool,
            key=f"painter_{st.session_state.canvas_key_id}_{'rect' if is_rect_tool else 'dot'}",
        )

    # The component reports the click in the pixels of the image as actually shown
    # (click["width"] x click["height"]), so dividing by that size gives the exact
    # position on the image no matter how the browser scaled it.
    if click is not None and click.get("unix_time") != st.session_state.last_click_time:
        st.session_state.last_click_time = click.get("unix_time")
        shown_w = max(float(click.get("width") or disp_w) - 1.0, 1.0)
        shown_h = max(float(click.get("height") or disp_h) - 1.0, 1.0)

        def to_norm(value, size):
            return min(max(float(value) / size, 0.0), 1.0)

        if is_rect_tool and "x1" in click:
            st.session_state.mask_strokes.append({
                "tool": "rect", "erase": False,
                "x1": to_norm(click["x1"], shown_w), "y1": to_norm(click["y1"], shown_h),
                "x2": to_norm(click["x2"], shown_w), "y2": to_norm(click["y2"], shown_h),
            })
            safe_rerun()
        elif not is_rect_tool and "x" in click:
            st.session_state.mask_strokes.append({
                "tool": "dot", "erase": is_eraser,
                "x": to_norm(click["x"], shown_w), "y": to_norm(click["y"], shown_h),
                "r": (brush_size / 2.0) / float(disp_w),
            })
            safe_rerun()

    st.caption(f"Strokes painted: {len(st.session_state.mask_strokes)}  (use ↩️ Undo or 🧹 Clear Mask in the sidebar)")

    # Build Binary Mask directly at Processing Resolution
    raw_mask_proc = build_mask_from_strokes(st.session_state.mask_strokes, target_shape=(proc_h, proc_w))

    # Apply Mask Dilation
    dilated_mask_proc = dilate_mask(raw_mask_proc, dilation_radius=mask_dilation)

    # Show mask overlay preview below the canvas (compact, inside a half-width column)
    st.subheader("Red Mask Overlay Preview")
    st.caption("Verifies exact pixel alignment of mask over the processing image.")
    prev_col, _ = st.columns([1, 1])
    with prev_col:
        if np.count_nonzero(dilated_mask_proc) > 0:
            overlay_bgr = create_mask_overlay(proc_bgr, dilated_mask_proc, alpha=0.45)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Mask Alignment Preview", use_column_width=True)
        else:
            st.image(np.array(disp_pil), caption="Awaiting mask stroke...", use_column_width=True)

    # ---------------------------------------------------------
    # STEP 3: RUN INPAINTING & DISPLAY RESULTS
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("Step 3: Inpainting Results & Quality Metrics")

    run_btn = st.button("🚀 Run Inpainting Pipeline", type="primary", use_container_width=True)

    if run_btn:
        # Validation: Ensure non-empty mask
        if np.count_nonzero(dilated_mask_proc) == 0:
            st.warning("⚠️ No mask detected! Please paint over the region to remove using the canvas above.")
            return

        masked_count = np.count_nonzero(dilated_mask_proc)
        total_pixels = proc_h * proc_w
        mask_coverage_pct = (masked_count / float(total_pixels)) * 100.0

        if mask_coverage_pct > 80.0:
            st.error("⚠️ Mask covers more than 80% of the image! Please select a smaller target area.")
            return

        logger.info(f"Starting inpainting run. Method: {method}, Coverage: {mask_coverage_pct:.2f}%")

        results = {}
        metrics_summary = []

        # Helper wrapper for LaMa model
        def run_lama_core(img, msk):
            return inpaint_lama(img, msk)

        # Helper wrapper for Telea
        def run_telea_core(img, msk):
            return inpaint_telea(img, msk, radius=telea_radius)

        # Execute OpenCV Telea
        if method == "OpenCV Telea" or method == "Both (Compare Side-by-Side)":
            with st.spinner("⏳ Executing OpenCV Telea Fast Marching algorithm..."):
                t0 = time.perf_counter()
                if use_smart_crop:
                    telea_res = smart_crop_inpaint(proc_bgr, dilated_mask_proc, run_telea_core)
                else:
                    telea_res = run_telea_core(proc_bgr, dilated_mask_proc)
                t_telea = time.perf_counter() - t0

                metrics_t = calculate_unmasked_metrics(proc_bgr, telea_res, dilated_mask_proc)
                results["Telea"] = (telea_res, t_telea, metrics_t)
                metrics_summary.append({
                    "Method": "OpenCV Telea",
                    "Execution Time (s)": f"{t_telea:.3f}",
                    "Resolution": f"{proc_w}x{proc_h}",
                    "Mask Coverage": f"{mask_coverage_pct:.2f}%",
                    "Device": "CPU",
                    "Unmasked PSNR (dB)": f"{metrics_t['psnr']} dB",
                    "Unmasked SSIM": f"{metrics_t['ssim']}"
                })

        # Execute Deep Learning LaMa
        if method == "LaMa AI Model" or method == "Both (Compare Side-by-Side)":
            with st.spinner("⏳ Executing LaMa Deep Learning Neural Network..."):
                # Load cached model
                _ = load_cached_lama_model()
                t0 = time.perf_counter()
                try:
                    if use_smart_crop:
                        lama_res = smart_crop_inpaint(proc_bgr, dilated_mask_proc, run_lama_core)
                    else:
                        lama_res = run_lama_core(proc_bgr, dilated_mask_proc)
                except Exception as ex:
                    st.error(f"❌ Error during LaMa execution: {ex}. Falling back to standard pipeline...")
                    lama_res = run_lama_core(proc_bgr, dilated_mask_proc)

                t_lama = time.perf_counter() - t0
                metrics_l = calculate_unmasked_metrics(proc_bgr, lama_res, dilated_mask_proc)
                results["LaMa"] = (lama_res, t_lama, metrics_l)
                metrics_summary.append({
                    "Method": "LaMa AI Model",
                    "Execution Time (s)": f"{t_lama:.3f}",
                    "Resolution": f"{proc_w}x{proc_h}",
                    "Mask Coverage": f"{mask_coverage_pct:.2f}%",
                    "Device": str(active_device).upper(),
                    "Unmasked PSNR (dB)": f"{metrics_l['psnr']} dB",
                    "Unmasked SSIM": f"{metrics_l['ssim']}"
                })

        # ---------------------------------------------------------
        # DISPLAY FULL OUTPUT IMAGES
        # ---------------------------------------------------------
        st.subheader("Full Output Images")
        num_cols = len(results) + 2
        res_cols = st.columns(num_cols)

        # Col 0: Original
        with res_cols[0]:
            st.image(np.array(proc_pil), caption="Original Image", use_column_width=True)
            st.download_button(
                "⬇️ Download Original",
                data=convert_cv_to_bytes(proc_bgr),
                file_name="original_image.png",
                mime="image/png"
            )

        # Col 1: Binary Mask
        with res_cols[1]:
            st.image(dilated_mask_proc, caption=f"Dilated Mask ({mask_coverage_pct:.1f}%)", use_column_width=True)
            st.download_button(
                "⬇️ Download Mask",
                data=convert_cv_to_bytes(dilated_mask_proc),
                file_name="binary_mask.png",
                mime="image/png"
            )

        # Col 2+: Results
        col_idx = 2
        for name, (out_bgr, p_time, m_dict) in results.items():
            with res_cols[col_idx]:
                out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                st.image(out_rgb, caption=f"{name} Result ({p_time:.3f} s)", use_column_width=True)
                st.download_button(
                    f"⬇️ Download {name}",
                    data=convert_cv_to_bytes(out_bgr),
                    file_name=f"inpainted_{name.lower()}.png",
                    mime="image/png"
                )
            col_idx += 1

        # ---------------------------------------------------------
        # ZOOMED DETAIL INSPECTION PANEL
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🔎 Zoomed Inpainting Detail Inspection")
        st.caption("Close-up view centered on the mask region to compare restoration fidelity.")

        zoom_cols = st.columns(len(results) + 1)
        
        # Zoomed Original
        orig_zoom = get_zoomed_crop(proc_bgr, dilated_mask_proc, crop_size=240)
        with zoom_cols[0]:
            st.image(cv2.cvtColor(orig_zoom, cv2.COLOR_BGR2RGB), caption="Zoomed Original (Corrupted)", use_column_width=True)

        # Zoomed Results
        z_idx = 1
        for name, (out_bgr, _, _) in results.items():
            res_zoom = get_zoomed_crop(out_bgr, dilated_mask_proc, crop_size=240)
            with zoom_cols[z_idx]:
                st.image(cv2.cvtColor(res_zoom, cv2.COLOR_BGR2RGB), caption=f"Zoomed {name}", use_column_width=True)
            z_idx += 1

        # ---------------------------------------------------------
        # METRICS & SANITY CHECK TABLE
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Performance & Fidelity Metrics")
        st.table(metrics_summary)
        
        st.caption(
            "💡 **Understanding Metrics**: "
            "**PSNR** (Peak Signal-to-Noise Ratio in dB) and **SSIM** (Structural Similarity Index from 0 to 1) "
            "are calculated **only on the unmasked (unaltered) background pixels**. "
            "High PSNR (>50 dB) and SSIM (~1.00) confirm that background pixels outside the mask were preserved with 100% pixel-perfect fidelity."
        )

        st.success("✅ Inpainting pipeline execution finished successfully!")

    # ---------------------------------------------------------
    # ABOUT EXPANDER (VIVA EXAM PREPARATION)
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("📚 About OpenCV Telea vs LaMa AI Model (Viva Exam Guide)"):
        st.markdown("""
        ### Algorithm Comparison & Project Background
        
        * **OpenCV Telea Inpainting**:
          * **Type**: Traditional non-learning Computer Vision algorithm (Fast Marching Method, proposed by Alexandru Telea in 2004).
          * **How it works**: It starts at the boundary of the mask and propagates color gradients inward using a weighted average of neighboring pixels within a specified radius.
          * **Pros & Cons**: Extremely fast (<0.05 seconds), lightweight, no GPU required. However, it struggles with complex textures or large missing structures, creating blurry or smooth patches.
          
        * **LaMa (Large Mask Inpainting)**:
          * **Type**: Deep Learning Neural Network (Fast Fourier Convolutions - FFC architecture).
          * **How it works**: Uses a pre-trained deep convolutional neural network that understands semantic shapes, scene geometry, and high-frequency textures across the entire image context.
          * **Pros & Cons**: Exceptional quality on complex structural backgrounds and large masks. Takes slightly longer to execute (~1-3 seconds on CPU/GPU) and requires deep learning weights (`big-lama.pt`).
        """)

if __name__ == "__main__":
    main()
