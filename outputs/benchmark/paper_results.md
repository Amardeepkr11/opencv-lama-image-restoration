# Benchmark: OpenCV Telea vs LaMa for Image Inpainting

All numbers below come from `results_all.csv` (72 tests per method) produced by `benchmark.py`
in this folder. Nothing was estimated or copied from other papers.

## 1. Experimental setup

| Item | Value |
|---|---|
| Dataset | Kodak Lossless True Color Image Suite, all 24 images (`kodim01`–`kodim24`), downloaded from r0k.us |
| Resolution | Each image resized (INTER_AREA) so the long side is 512 px: 512×341 (landscape) or 341×512 (portrait) |
| Hole sizes | 5%, 15%, 30% of the image area; measured mean coverage 4.97%, 14.87%, 29.68% |
| Mask shape | Random cluster of filled circles, filled ellipses and thick poly-line strokes around one random centre (centre drawn from the middle 40% of the image) |
| Reproducibility | Fixed seed 42; each mask uses its own generator seeded by (42, image index, hole size), so any subset reproduces the same masks |
| Tests | 24 images × 3 hole sizes = 72 tests per method |
| Methods | OpenCV Telea (`cv2.inpaint`, `INPAINT_TELEA`, radius 3) via `utils/telea.py`; LaMa (Big-LaMa TorchScript, `simple-lama-inpainting` weights) via `utils/lama.py` |
| Input to methods | Image with hole pixels set to 0, plus the binary mask |
| Ground truth | The original (resized) photo |
| Metrics | PSNR and SSIM inside the hole only; PSNR and SSIM over the full image; wall-clock time per image (`time.perf_counter`) |
| SSIM details | scikit-image 0.19.3, 7×7 window, per-channel map averaged over RGB; "hole SSIM" = mean of the SSIM map over hole pixels |
| Timing | LaMa loaded and warmed up twice on a dummy 512×512 image before timing; time includes only the inpainting call |
| Hardware | CPU-only laptop: Intel Core i5-11320H (4 cores / 8 threads, 3.2 GHz), 7.8 GB RAM, Windows 11; PyTorch 1.13.1 (CPU, 4 threads), OpenCV 5.0.0, Python 3.7.9 |

Sanity check: in all 144 runs, the number of pixels changed outside the hole was 0 for both
methods (`changed_outside_px` column), so differences come only from the filled region.

Note on the LaMa wrapper: during this work a bug in `utils/lama.py` was fixed. LaMa pads its input
to a multiple of 8, and the wrapper used to *resize* the padded output back to the input size,
which shifts every pixel (outside-hole PSNR dropped to 21.4 dB on a 512×341 test image). The
wrapper now crops the padding. All results here use the fixed version.

## 2. Results

### 2.1 Overall (mean ± standard deviation, 72 tests per method)

| Method | Tests | PSNR hole (dB) | SSIM hole | PSNR full (dB) | SSIM full | Time / image (s) |
|---|---|---|---|---|---|---|
| Telea | 72 | 18.15 ± 3.18 | 0.471 ± 0.150 | 27.01 ± 5.24 | 0.9084 ± 0.0635 | 0.041 ± 0.024 |
| LaMa | 72 | 19.80 ± 3.59 | 0.522 ± 0.165 | 28.67 ± 5.48 | 0.9152 ± 0.0628 | 3.524 ± 0.306 |

### 2.2 By hole size (mean ± standard deviation, 24 tests per row)

| Hole (target / actual mean) | Method | PSNR hole (dB) | SSIM hole | PSNR full (dB) | SSIM full | Time / image (s) |
|---|---|---|---|---|---|---|
| 5% / 4.97% | Telea | 19.51 ± 3.77 | 0.499 ± 0.167 | 32.55 ± 3.78 | 0.9735 ± 0.0089 | 0.015 ± 0.002 |
| 5% / 4.97% | LaMa | 21.20 ± 4.28 | 0.575 ± 0.184 | 34.24 ± 4.29 | 0.9775 ± 0.0097 | 3.456 ± 0.294 |
| 15% / 14.87% | Telea | 18.05 ± 2.99 | 0.455 ± 0.143 | 26.33 ± 2.99 | 0.9161 ± 0.0223 | 0.039 ± 0.008 |
| 15% / 14.87% | LaMa | 19.50 ± 3.60 | 0.508 ± 0.153 | 27.78 ± 3.60 | 0.9243 ± 0.0239 | 3.519 ± 0.304 |
| 30% / 29.68% | Telea | 16.88 ± 2.10 | 0.458 ± 0.142 | 22.15 ± 2.09 | 0.8357 ± 0.0434 | 0.069 ± 0.012 |
| 30% / 29.68% | LaMa | 18.70 ± 2.28 | 0.484 ± 0.149 | 23.98 ± 2.28 | 0.8439 ± 0.0456 | 3.597 ± 0.316 |

### 2.3 Paired comparison (same image, same mask)

Δ = LaMa − Telea. Wilcoxon signed-rank test, two-sided, paired over images.

| Hole | Tests | LaMa wins (PSNR hole) | LaMa wins (SSIM hole) | Mean ΔPSNR hole (dB) | Median ΔPSNR | Mean ΔSSIM hole | Wilcoxon p (PSNR) | Wilcoxon p (SSIM) | LaMa / Telea time |
|---|---|---|---|---|---|---|---|---|---|
| 5% | 24 | 19 / 24 | 20 / 24 | +1.70 | +1.97 | +0.076 | 1.5e-04 | 1.1e-04 | 234× |
| 15% | 24 | 21 / 24 | 18 / 24 | +1.45 | +1.50 | +0.054 | 2.5e-05 | 2.8e-04 | 93× |
| 30% | 24 | 22 / 24 | 16 / 24 | +1.82 | +1.56 | +0.026 | 8.3e-07 | 7.9e-02 | 54× |
| All | 72 | 62 / 72 | 54 / 72 | +1.66 | +1.56 | +0.052 | 4.0e-11 | 1.1e-07 | 127× |

**LaMa beat Telea in 62 of 72 tests on hole PSNR, and in 54 of 72 tests on hole SSIM.**

Charts: `chart_psnr.png`, `chart_ssim.png`, `chart_time.png`. Per-test images: `examples/<image>_hole<size>.png`.

## 3. Analysis

**Quality.** LaMa had higher hole PSNR than Telea at every hole size, by about 1.5–1.8 dB on
average (+1.66 dB overall). The paired PSNR difference is statistically significant at all three
sizes (p ≤ 1.5e-04). On hole SSIM LaMa was also better on average at every size, but the margin
shrinks as the hole grows (+0.076 at 5%, +0.054 at 15%, +0.026 at 30%), and at 30% the SSIM
difference is not significant (p = 0.079; LaMa higher in 16 of 24 tests). So at large holes LaMa
reliably gets the colours/intensities closer (PSNR), but its advantage in local structure (SSIM)
is small.

**Per image.** LaMa won on PSNR at all three hole sizes for 17 of the 24 images. Only two images had
a negative average PSNR difference across their three tests: kodim07 (−0.10 dB) and kodim17
(−0.09 dB). The largest LaMa gains were on kodim20 (+5.63 dB at 30%, +5.27 dB at 5%) and kodim10
(+4.57 dB at 30%, +4.55 dB at 15%).

**When is Telea enough?** In this benchmark there was no hole size at which Telea matched LaMa on
average: even at the smallest tested hole (5%), LaMa won 19 of 24 tests on PSNR and 20 of 24 on
SSIM. Telea was better in 10 of the 72 tests on PSNR: 5 at 5%, 3 at 15% and 2 at 30%. Telea wins
slightly more often on the smallest holes, but even there it loses 19 of 24 tests, and its average
PSNR deficit at 5% (−1.70 dB) is about the same as at 30% (−1.82 dB). The data therefore does not
show a hole size where Telea is "good enough" in quality; Telea's case rests on speed.

**Speed.** Telea took 0.015–0.069 s per image on average (growing with hole size), LaMa
3.46–3.60 s (almost independent of hole size, since the network always processes the whole
image). On average LaMa was 127× slower per test (234× at 5%, 54× at 30%). Model loading and
warm-up are excluded from these times.

**Full-image metrics** follow the hole metrics, because both methods leave all pixels outside the
hole unchanged; they mainly fall as the hole gets bigger (e.g. full PSNR for LaMa: 34.24 dB at 5%,
23.98 dB at 30%).

## 4. Limitations

1. **Synthetic random masks.** Holes are random blobs and strokes, not real objects. Real object
   removal has object-shaped masks and the true background behind the object is unknown; here the
   ground truth is the pixels that were cut out, which is only a proxy for that task.
2. **PSNR/SSIM measure similarity to the original, not realism.** A blurry fill can score well and a
   sharp but different (yet plausible) fill can score badly. Example in kodim07 at 15% below: LaMa
   has lower PSNR but much higher SSIM and visibly reconstructs the shutters. No perceptual metric
   (e.g. LPIPS, FID) or user study was done.
3. **Hole SSIM is computed with a 7×7 window**, so values at pixels near the hole border also
   depend on the known pixels around them.
4. **Only 512 px images.** Results at the app's processing size (up to 2048 px) were not measured;
   LaMa's time will grow with image size.
5. **CPU only, one laptop.** Times are from a single 4-core CPU run; GPU timing was not measured.
   Each test was run once, so timing variance between runs is not captured.
6. **Small dataset.** 24 images, one mask per image and hole size, one seed.
7. **Smallest hole is 5% (about 8,700 px at 512×341).** Thin scratches or text lines, which are the
   classical use case for Telea, were not tested.
8. **One Telea setting** (radius 3) and one LaMa model (Big-LaMa) were used, without tuning.

## 5. Failure cases

Each case refers to the image file in `examples/` (columns: Original | Masked | Telea | LaMa).

1. **`kodim12_hole5.png` — LaMa loses on both metrics on a smooth area.** The hole lies on the wet
   sand / water edge of the beach. Telea: 28.79 dB / SSIM 0.606, LaMa: 27.17 dB / SSIM 0.482, the
   largest LaMa loss in the benchmark (−1.63 dB, −0.124 SSIM). Visually, Telea leaves a pale
   smudge in the upper part of the hole while LaMa continues the sand texture, yet the metrics
   prefer Telea's smooth fill; this case shows the metric limitation from Section 4.2 as much as a
   LaMa error.
2. **`kodim07_hole15.png` — metrics disagree.** The hole covers the shuttered window and the pink
   flower. Telea (18.03 dB / 0.355) produces flat, faceted patches across the shutters; LaMa
   (17.38 dB / 0.508) redraws shutter slats and leaves. PSNR prefers Telea, SSIM prefers LaMa by
   +0.154.
3. **`kodim20_hole30.png` — Telea fails on structure.** The hole covers the aircraft fuselage and
   part of the sky. Telea (16.44 dB / 0.757) smears the fuselage into grey streaks and leaves a
   grey artefact in the sky; LaMa (22.06 dB / 0.843) continues the fuselage outline. This is the
   largest gap in the benchmark (+5.63 dB for LaMa).
4. **`kodim05_hole15.png` — both fail on a cluttered scene.** The hole covers overlapping riders and
   motorbikes. Telea (13.26 dB / 0.195) fills it with coloured blotches; LaMa (13.72 dB / 0.191)
   produces a blurry mix of bike parts. Neither recovers the scene, and these are among the lowest
   scores of the benchmark (kodim05 and kodim08 at 15–30% are the hardest cases for both methods).

## 6. Hybrid strategy (Telea for small holes, LaMa for large ones)

The numbers **do not support** a size-based hybrid on quality grounds. LaMa was better on average
at every tested hole size, including the smallest (5%: +1.70 dB, 19 of 24 wins on PSNR). Telea
won slightly more tests at 5% (5) than at 30% (2), which is a weak hint in the hybrid's direction,
but far from enough to prefer Telea at 5%.

What the data does support is a speed/quality trade-off: Telea is 54–234× faster, so it remains a
reasonable fast preview or a fallback when LaMa is unavailable, at the cost of about 1.5–1.8 dB
hole PSNR on this benchmark. Whether Telea is sufficient for very thin masks (scratches, text) is
an open question that this benchmark did not test; a follow-up experiment with thin-stroke masks
would be needed before recommending a size-based switch.
