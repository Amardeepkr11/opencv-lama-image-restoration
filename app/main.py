import os
import time
import hashlib
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

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton button, .stDownloadButton button {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.block-container { padding: 1.4rem 2rem 2rem 2rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Header */
.app-hero {
    border: 1px solid #1F2A3D; border-radius: 14px; padding: 18px 22px;
    background: linear-gradient(135deg, rgba(45,212,191,0.10), rgba(21,30,46,0.6) 55%);
    margin-bottom: 10px;
}
.app-hero h1 { font-size: 1.65rem; font-weight: 700; margin: 0 0 2px 0; padding: 0; letter-spacing: -0.01em; }
.app-hero p { color: #94A3B8; margin: 0 0 10px 0; font-size: 0.95rem; }
.badge {
    display: inline-block; padding: 2px 10px; margin-right: 6px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; border: 1px solid #2B3A52; color: #CBD5E1; background: #111A2A;
}
.badge.accent { border-color: rgba(45,212,191,0.5); color: #5EEAD4; }

/* Step headings */
.step { display: flex; align-items: center; gap: 10px; margin: 18px 0 6px 0; }
.step-num {
    width: 26px; height: 26px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
    background: #2DD4BF; color: #0B1120; font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
}
.step-title { font-size: 1.1rem; font-weight: 600; }
.step-hint { color: #64748B; font-size: 0.85rem; }
.info-line { color: #94A3B8; font-size: 0.85rem; margin: 2px 0 0 0; }
.info-line b { color: #E2E8F0; font-weight: 600; }

/* Cards & widgets */
div[data-testid="metric-container"], div[data-testid="stMetric"] {
    background: #111A2A; border: 1px solid #1F2A3D; border-radius: 10px; padding: 8px 12px;
}
div[data-testid="stMetricLabel"] { color: #94A3B8; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; }
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 600; border: 1px solid #2B3A52;
}
.stButton > button[kind="primary"] { border: none; color: #0B1120; min-height: 2.9rem; font-size: 1rem; }
[data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stImage"] [data-testid="caption"], .stImage div { color: #94A3B8; }
iframe[title*="brush_canvas"], iframe[title*="image_coordinates"] { border-radius: 10px; }
[data-testid="stFileUploader"] section { padding: 10px 14px; border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { padding: 6px 14px; border-radius: 8px 8px 0 0; }
div[data-testid="stExpander"] { border-radius: 10px; }
.result-label { font-weight: 600; font-size: 0.95rem; margin: 0 0 4px 0; }

/* Sidebar */
section[data-testid="stSidebar"] div:has(> .block-container) { padding-top: 2.6rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.6rem; }
.sb-title { font-size: 1.05rem; font-weight: 700; margin: 0 0 2px 0; }
.sb-sub { color: #64748B; font-size: 0.8rem; margin: 0 0 8px 0; }
</style>
"""


def step_header(num: int, title: str, hint: str = ""):
    hint_html = f'<span class="step-hint">{hint}</span>' if hint else ""
    st.markdown(
        f'<div class="step"><span class="step-num">{num}</span>'
        f'<span class="step-title">{title}</span>{hint_html}</div>',
        unsafe_allow_html=True,
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
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-hero">
          <h1>AI-Based Image Inpainting System</h1>
          <p>Object &amp; text removal using OpenCV Telea and LaMa</p>
          <span class="badge">OpenCV Telea</span><span class="badge">LaMa</span><span class="badge accent">SDG 9</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # SIDEBAR: three main controls + collapsed advanced settings
    # ---------------------------------------------------------
    st.sidebar.markdown('<div class="sb-title">Inpainting controls</div>'
                        '<div class="sb-sub">Pick a method and paint over what to remove.</div>',
                        unsafe_allow_html=True)

    method = st.sidebar.radio("Method", ["Telea", "LaMa", "Both"], index=2, horizontal=True)
    run_telea_method = method in ("Telea", "Both")
    run_lama_method = method in ("LaMa", "Both")

    tool_choice = st.sidebar.selectbox(
        "Drawing tool",
        ["Freehand Brush (drag)", "Freehand Eraser (drag)",
         "Brush Dot (click)", "Eraser Dot (click)", "Rectangle Box (drag)"],
        index=0
    )
    is_freehand = "Freehand" in tool_choice
    is_rect_tool = "Rectangle" in tool_choice
    is_eraser = "Eraser" in tool_choice

    brush_size = st.sidebar.slider("Brush size (px)", min_value=5, max_value=120, value=25)

    active_device = get_active_device()
    with st.sidebar.expander("Advanced settings", expanded=False):
        mask_dilation = st.slider("Mask dilation (px)", min_value=0, max_value=15, value=3,
                                  help="Slightly expands mask boundaries to prevent edge halos or residual outlines.")
        telea_radius = float(st.slider("Telea radius", min_value=1.0, max_value=15.0, value=3.0))
        max_proc_size = st.select_slider(
            "Max image size (px)",
            options=[512, 768, 1024, 1280, 1536, 2048],
            value=1024,
            help="Limits max image side during inpainting processing to prevent memory issues."
        )
        use_smart_crop = st.checkbox(
            "Smart crop around mask",
            value=True,
            help="Crops region around mask for up to 5x faster LaMa inference, then soft-blends back into full image."
        )
        show_alignment_preview = st.checkbox("Show mask alignment preview", value=False,
                                             help="Shows the dilated mask on the processing-size image.")
        device_label = "GPU (CUDA)" if active_device.type == "cuda" else "CPU"
        st.caption(f"Compute device: {device_label}")

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

    # ---------------------------------------------------------
    # STEP 1: UPLOAD
    # ---------------------------------------------------------
    step_header(1, "Upload an image", "JPG or PNG, any size")
    up_col, opt_col = st.columns([3, 1])
    with up_col:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with opt_col:
        use_sample = st.checkbox("Use demo image", value=uploaded_file is None)

    pil_image_raw = None
    if uploaded_file is not None:
        try:
            pil_image_raw = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load uploaded image: {e}")
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
            st.error(f"Error loading sample image: {e}")
            return

    if pil_image_raw is None:
        st.info("Upload an image above, or tick 'Use demo image'.")
        return

    # Resize raw image to processing size (max_proc_size) preserving aspect ratio
    proc_pil, (orig_w, orig_h) = resize_image_aspect(pil_image_raw, max_dim=max_proc_size)
    proc_w, proc_h = proc_pil.size
    proc_bgr = cv2.cvtColor(np.array(proc_pil), cv2.COLOR_RGB2BGR)

    # Resize for UI Canvas display: at most 700 px wide and 520 px tall so the
    # canvas and its controls fit on a laptop screen
    disp_scale = min(700.0 / proc_w, 520.0 / proc_h, 1.0)
    disp_w, disp_h = max(1, int(proc_w * disp_scale)), max(1, int(proc_h * disp_scale))
    disp_pil = proc_pil.resize((disp_w, disp_h), Image.LANCZOS) if disp_scale < 1.0 else proc_pil

    # A new image invalidates the strokes painted on the previous one
    image_signature = (uploaded_file.name if uploaded_file is not None else "sample", orig_w, orig_h)
    if st.session_state.get("image_signature") != image_signature:
        st.session_state.image_signature = image_signature
        st.session_state.mask_strokes = []
        st.session_state.last_click_time = None
        st.session_state.canvas_key_id += 1

    image_name = uploaded_file.name if uploaded_file is not None else "Demo image"
    st.markdown(
        f'<p class="info-line"><b>{image_name}</b> &nbsp;·&nbsp; {orig_w} × {orig_h} px'
        f' &nbsp;·&nbsp; processed at {proc_w} × {proc_h} px</p>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # STEP 2: MARK THE AREA
    # ---------------------------------------------------------
    if is_freehand and is_eraser:
        tool_hint = "Hold the mouse button and drag to erase parts of the red mask."
    elif is_freehand:
        tool_hint = "Hold the mouse button and drag to paint red over the object or text to remove."
    elif is_rect_tool:
        tool_hint = "Press, drag and release over the region you want to remove."
    else:
        tool_hint = "Click on the image to paint red dots over the region you want to remove."
    step_header(2, "Mark the area", tool_hint)

    canvas_col, side_col = st.columns([3, 1])

    # Show the display image with the current mask painted on top of it
    disp_mask = build_mask_from_strokes(st.session_state.mask_strokes, target_shape=(disp_h, disp_w))
    disp_rgb = np.array(disp_pil)
    disp_overlay_rgb = cv2.cvtColor(
        create_mask_overlay(cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR), disp_mask, alpha=0.55),
        cv2.COLOR_BGR2RGB
    )

    if is_freehand:
        with canvas_col:
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
        with canvas_col:
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

    # Build Binary Mask directly at Processing Resolution
    raw_mask_proc = build_mask_from_strokes(st.session_state.mask_strokes, target_shape=(proc_h, proc_w))

    # Apply Mask Dilation
    dilated_mask_proc = dilate_mask(raw_mask_proc, dilation_radius=mask_dilation)

    masked_count = np.count_nonzero(dilated_mask_proc)
    total_pixels = proc_h * proc_w
    mask_coverage_pct = (masked_count / float(total_pixels)) * 100.0

    with side_col:
        st.metric("Mask coverage", f"{mask_coverage_pct:.1f}%")
        st.caption(f"Strokes painted: {len(st.session_state.mask_strokes)}")
        st.button("Undo", use_container_width=True, on_click=undo_last_stroke)
        st.button("Clear Mask", use_container_width=True, on_click=clear_mask)
        st.caption(f"Method: {method} · Tool: {tool_choice.split(' (')[0]}")
    with canvas_col:
        run_btn = st.button("Remove Object", type="primary", use_container_width=True)

    if show_alignment_preview:
        prev_col, _ = st.columns([1, 1])
        with prev_col:
            if masked_count > 0:
                overlay_bgr = create_mask_overlay(proc_bgr, dilated_mask_proc, alpha=0.45)
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, caption="Mask Alignment Preview", use_column_width=True)
            else:
                st.image(np.array(disp_pil), caption="Awaiting mask stroke...", use_column_width=True)

    # ---------------------------------------------------------
    # STEP 3: RESULTS
    # ---------------------------------------------------------
    step_header(3, "Results", "Original vs inpainted, with timing and downloads")

    # Results stay visible across reruns (e.g. download clicks) while the mask and settings are unchanged
    run_signature = (
        image_signature, hashlib.md5(dilated_mask_proc.tobytes()).hexdigest(),
        method, telea_radius, max_proc_size, use_smart_crop,
    )

    if run_btn:
        # Validation: Ensure non-empty mask
        if masked_count == 0:
            st.warning("No mask yet. Paint over the region to remove in step 2 first.")
            return

        if mask_coverage_pct > 80.0:
            st.error("Mask covers more than 80% of the image. Please select a smaller target area.")
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
        if run_telea_method:
            with st.spinner("Running OpenCV Telea..."):
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
        if run_lama_method:
            with st.spinner("Running LaMa (deep learning)..."):
                # Load cached model
                _ = load_cached_lama_model()
                t0 = time.perf_counter()
                try:
                    if use_smart_crop:
                        lama_res = smart_crop_inpaint(proc_bgr, dilated_mask_proc, run_lama_core)
                    else:
                        lama_res = run_lama_core(proc_bgr, dilated_mask_proc)
                except Exception as ex:
                    st.error(f"Error during LaMa execution: {ex}. Falling back to standard pipeline...")
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

        st.session_state.last_results = {
            "signature": run_signature,
            "results": results,
            "metrics_summary": metrics_summary,
            "coverage": mask_coverage_pct,
            "mask": dilated_mask_proc.copy(),
            "proc_bgr": proc_bgr.copy(),
        }

    saved = st.session_state.get("last_results")
    if saved is None:
        st.caption("Paint a mask and click **Remove Object** to see the results here.")
    elif saved["signature"] != run_signature:
        st.caption("The mask or settings changed. Click **Remove Object** again to update the results.")
    else:
        results = saved["results"]
        res_mask = saved["mask"]
        res_orig_bgr = saved["proc_bgr"]
        res_h, res_w = res_orig_bgr.shape[:2]

        tab_compare, tab_zoom, tab_details = st.tabs(["Side by side", "Before / After zoom", "Details"])

        with tab_compare:
            cols = st.columns(len(results) + 1)
            with cols[0]:
                st.markdown('<p class="result-label">Original</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(res_orig_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.metric("Resolution", f"{res_w} × {res_h}")
                st.download_button("Download original", data=convert_cv_to_bytes(res_orig_bgr),
                                   file_name="original_image.png", mime="image/png", use_container_width=True)
            for col, (name, (out_bgr, p_time, _)) in zip(cols[1:], results.items()):
                with col:
                    st.markdown(f'<p class="result-label">{name}</p>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.metric("Time", f"{p_time:.2f} s")
                    st.download_button(f"Download {name}", data=convert_cv_to_bytes(out_bgr),
                                       file_name=f"inpainted_{name.lower()}.png", mime="image/png",
                                       use_container_width=True)

        with tab_zoom:
            st.caption("Close-up of the edited area. Before shows the mask in red.")
            zoom_cols = st.columns(len(results) + 1)
            before_bgr = create_mask_overlay(res_orig_bgr, res_mask, alpha=0.45)
            with zoom_cols[0]:
                st.markdown('<p class="result-label">Before</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(get_zoomed_crop(before_bgr, res_mask, crop_size=240), cv2.COLOR_BGR2RGB),
                         use_column_width=True)
            for col, (name, (out_bgr, _, _)) in zip(zoom_cols[1:], results.items()):
                with col:
                    st.markdown(f'<p class="result-label">After · {name}</p>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(get_zoomed_crop(out_bgr, res_mask, crop_size=240), cv2.COLOR_BGR2RGB),
                             use_column_width=True)

        with tab_details:
            st.table(saved["metrics_summary"])
            st.caption(
                "PSNR and SSIM here are measured only on the pixels outside the mask, to confirm the "
                "background was preserved. High values (PSNR > 50 dB, SSIM ≈ 1.00) mean it is unchanged."
            )
            mask_col, _ = st.columns([1, 2])
            with mask_col:
                st.image(res_mask, caption=f"Dilated mask ({saved['coverage']:.1f}%)", use_column_width=True)
                st.download_button("Download mask", data=convert_cv_to_bytes(res_mask),
                                   file_name="binary_mask.png", mime="image/png", use_container_width=True)

    # ---------------------------------------------------------
    # ABOUT
    # ---------------------------------------------------------
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    with st.expander("About Telea vs LaMa"):
        st.markdown(
            "- **OpenCV Telea**: a classical algorithm (Fast Marching Method, 2004) that fills the hole from its "
            "edges inward using nearby pixels. Very fast, no GPU needed, but blurry on large or detailed areas.\n"
            "- **LaMa**: a deep neural network (Fast Fourier Convolutions) that uses context from the whole image "
            "to draw plausible texture and structure. Better quality on big holes, but slower on CPU."
        )

if __name__ == "__main__":
    main()
