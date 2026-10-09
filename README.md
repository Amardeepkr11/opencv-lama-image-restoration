# AI-Based Image Inpainting System using OpenCV and LaMa

A B.Tech Capstone Project implementing a hybrid Image Inpainting System combining traditional Computer Vision (**OpenCV Telea**) and state-of-the-art Deep Learning (**LaMa - Large Mask Inpainting**).

---

## 🌟 Key Features

* **Dual Inpainting Engine**:
  * **OpenCV Telea**: Fast Marching Method (FMM) algorithm for quick, lightweight repair of small scratches and thin lines.
  * **LaMa AI Model**: Fast Fourier Convolution (FFC) deep learning model for restoring complex structural backgrounds and large missing regions.
* **Pixel-Aligned Interactive Canvas**:
  * Draw freehand masks, rectangular boxes, or use the eraser tool directly over target objects/text.
  * Adjust brush size and mask dilation/expansion to eliminate edge halos.
  * Live red mask overlay preview for verifying alignment before running inpainting.
* **Smart Bounding-Box Crop Acceleration**:
  * Automatically crops the localized mask region with padded borders for up to **5x faster inference** on CPU/GPU, blending seamless feathered edges back into the full image.
* **Quality Metrics & Sanity Check**:
  * Calculates **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) on unmasked background regions to verify background preservation.
  * Zoomed detail inspection panel centered on the restored region.
* **Hardware Autodetect & Memory Safety**:
  * Automatically utilizes NVIDIA CUDA GPU acceleration if available, with smooth fallback to CPU mode.
  * Automatic image resizing for large images (e.g. 3000+ px) to prevent memory crashes.

---

## 📁 Folder Structure

```text
4th year/
├── app/
│   └── main.py          # Interactive Streamlit Web UI Application
├── models/              # Model weights storage folder
├── outputs/             # Output directory for test runs & generated comparisons
├── samples/             # Sample images directory
├── utils/
│   ├── __init__.py      # Package exports
│   ├── telea.py         # OpenCV Telea inpainting implementation
│   ├── lama.py          # LaMa model loader with CUDA/CPU fallback & caching
│   ├── image_io.py      # Aspect ratio resizing & bytes conversion helpers
│   ├── mask.py          # Mask extraction, dilation, overlay preview & smart cropping
│   └── metrics.py       # Unmasked PSNR & SSIM metrics calculator
├── venv/                # Python Virtual Environment
├── benchmark.py         # Telea vs LaMa benchmark (PSNR / SSIM / time)
├── get_photos.py        # Downloads the Kodak benchmark photos
├── compare_test.py      # Comparative test script
├── requirements.txt     # Pinned package dependencies
├── run_app.bat          # 1-Click launcher script for Windows
├── test_lama.py         # Phase 2 test script
├── test_telea.py        # Phase 1 test script
└── README.md            # Project documentation
```

---

## 🚀 Setup & Installation

1. **Clone / Open Workspace Folder**:
   Ensure you are in the project root folder.

2. **Virtual Environment Setup**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Requirements**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

### Method 1: 1-Click Launcher (Windows)
Double-click `run_app.bat` or execute in terminal:
```powershell
.\run_app.bat
```

### Method 2: Manual Terminal Command
```powershell
.\venv\Scripts\python.exe -m streamlit run app/main.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Test Scripts

You can execute standalone validation test scripts to verify backend algorithms without launching the web interface:

```powershell
# Phase 1: Test OpenCV Telea
.\venv\Scripts\python.exe test_telea.py

# Phase 2: Test LaMa Deep Learning Model
.\venv\Scripts\python.exe test_lama.py

# Comparative Test: Side-by-side output comparison
.\venv\Scripts\python.exe compare_test.py
```

Outputs are saved automatically into the `outputs/` directory.

---

## 📈 Benchmark (Telea vs LaMa)

A reproducible quality/speed comparison on the 24-image Kodak Lossless True Color set,
using the original photo as ground truth.

```powershell
# 1. Download the 24 Kodak photos into samples/benchmark/ (only needed once)
.\venv\Scripts\python.exe get_photos.py

# 2. Quick check, Telea only, first 3 photos (a few seconds)
.\venv\Scripts\python.exe benchmark.py --no-lama --limit 3 --out outputs/benchmark_quick

# 3. Full benchmark: 24 photos x 3 hole sizes, Telea + LaMa (about 5 min on a 4-core laptop CPU)
.\venv\Scripts\python.exe benchmark.py
```

Options: `--input` (photo folder), `--out` (result folder), `--max-dim` (long side in px, default 512),
`--coverages` (hole sizes in %, default `5,15,30`), `--seed` (default 42), `--no-lama`, `--limit N`, `--telea-radius`.

How it works: each photo is resized to 512 px (long side), a random hole (circles, ellipses and thick
strokes clustered around one point) is cut for each hole size with a fixed seed, both methods fill it,
and PSNR / SSIM are measured inside the hole and over the full image, plus the time per image.
LaMa is warmed up on a dummy image first so model loading is not counted.

Results are written to `outputs/benchmark/`:

| File | Content |
|---|---|
| `results_all.csv` | One row per photo x hole size x method |
| `summary_overall.csv` | Mean / std / median per method |
| `summary_by_hole_size.csv` | Same, split by hole size |
| `wins_by_hole_size.csv` | How often LaMa beat Telea, per hole size |
| `chart_psnr.png`, `chart_ssim.png`, `chart_time.png` | Bar charts (mean ± std) |
| `examples/` | Original \| Masked \| Telea \| LaMa strip for every test |
| `paper_results.md` | Write-up of the results for the research paper |

---

## ⚠️ Limitations

1. **Hardware Dependencies**: Deep learning inference on CPU takes ~1-3 seconds per image, whereas an NVIDIA GPU processes images in milliseconds.
2. **Extreme Mask Sizes**: Masks covering >80% of an image may lack sufficient surrounding structural context for accurate semantic reconstruction.
