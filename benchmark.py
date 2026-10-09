"""Telea vs LaMa inpainting benchmark on a folder of ground-truth photos.

For every photo and every hole size a reproducible random mask is generated, both
methods fill the hole, and the result is compared with the untouched original.

usage:
    .\\venv\\Scripts\\python.exe get_photos.py                 # download Kodak (once)
    .\\venv\\Scripts\\python.exe benchmark.py                  # full run (Telea + LaMa)
    .\\venv\\Scripts\\python.exe benchmark.py --no-lama --limit 3   # quick check
"""
import argparse
import glob
import logging
import math
import os
import platform
import time

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from utils.telea import inpaint_telea

METRICS = ["psnr_hole", "ssim_hole", "psnr_full", "ssim_full", "time_s"]
METRIC_LABELS = {
    "psnr_hole": "PSNR hole (dB)",
    "ssim_hole": "SSIM hole",
    "psnr_full": "PSNR full (dB)",
    "ssim_full": "SSIM full",
    "time_s": "Time (s)",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_image(path: str, max_dim: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def make_mask(h: int, w: int, coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Random hole of roughly `coverage` (0..1) of the image: a cluster of circles,
    ellipses and thick strokes around one random centre."""
    target = coverage * h * w
    mask = np.zeros((h, w), np.uint8)
    cx = rng.uniform(0.3, 0.7) * w
    cy = rng.uniform(0.3, 0.7) * h
    spread = math.sqrt(target / math.pi)
    size_scale = 1.0

    for _ in range(600):
        current = np.count_nonzero(mask)
        if current >= 0.97 * target:
            break
        s = math.sqrt(target) * rng.uniform(0.2, 0.45) * size_scale
        ang = rng.uniform(0, 2 * math.pi)
        dist = spread * math.sqrt(rng.uniform(0, 1))
        px, py = cx + dist * math.cos(ang), cy + dist * math.sin(ang)

        cand = mask.copy()
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(cand, (int(px), int(py)), max(2, int(s / 2)), 255, -1)
        elif kind == 1:
            axes = (max(2, int(s * rng.uniform(0.4, 0.8))), max(2, int(s * rng.uniform(0.2, 0.5))))
            cv2.ellipse(cand, (int(px), int(py)), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
        else:
            pts = [(px, py)]
            for _ in range(int(rng.integers(2, 5))):
                a = rng.uniform(0, 2 * math.pi)
                pts.append((pts[-1][0] + s * 0.7 * math.cos(a), pts[-1][1] + s * 0.7 * math.sin(a)))
            thickness = max(3, int(s * rng.uniform(0.12, 0.25)))
            cv2.polylines(cand, [np.array(pts, np.int32)], False, 255, thickness)

        if np.count_nonzero(cand) <= 1.05 * target:
            mask = cand
        else:
            size_scale *= 0.7
    return mask


# ---------------------------------------------------------------------------
# Metrics (original photo = ground truth)
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def compute_metrics(orig: np.ndarray, result: np.ndarray, mask: np.ndarray) -> dict:
    hole = mask > 0
    _, ssim_map = structural_similarity(orig, result, channel_axis=2, full=True, data_range=255)
    ssim_map = ssim_map.mean(axis=2)
    return {
        "psnr_hole": psnr(orig[hole], result[hole]),
        "ssim_hole": float(ssim_map[hole].mean()),
        "psnr_full": psnr(orig, result),
        "ssim_full": float(ssim_map.mean()),
        "changed_outside_px": int(np.count_nonzero(np.any(orig[~hole] != result[~hole], axis=1))),
    }


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------
def labelled(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def save_example(path, orig, mask, results: dict, rows: dict):
    masked = orig.copy()
    masked[mask > 0] = (255, 255, 255)
    panels = [labelled(orig, "Original"), labelled(masked, f"Masked ({100 * np.count_nonzero(mask) / mask.size:.1f}%)")]
    for name, res in results.items():
        r = rows[name]
        panels.append(labelled(res, f"{name}  {r['psnr_hole']:.2f} dB / SSIM {r['ssim_hole']:.3f}"))
    gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.hstack([strip, gap, p])
    cv2.imwrite(path, strip)


def save_charts(df: pd.DataFrame, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(df["method"].unique())
    sizes = sorted(df["hole_target_pct"].unique())
    colors = {"Telea": "#4C72B0", "LaMa": "#DD8452"}
    charts = {
        "chart_psnr.png": [("psnr_hole", "PSNR in hole (dB)"), ("psnr_full", "PSNR full image (dB)")],
        "chart_ssim.png": [("ssim_hole", "SSIM in hole"), ("ssim_full", "SSIM full image")],
        "chart_time.png": [("time_s", "Inpainting time per image (s)")],
    }
    for fname, panels in charts.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), squeeze=False)
        for ax, (metric, title) in zip(axes[0], panels):
            width = 0.8 / len(methods)
            x = np.arange(len(sizes))
            for i, m in enumerate(methods):
                g = df[df["method"] == m].groupby("hole_target_pct")[metric]
                mean, std = g.mean().reindex(sizes), g.std().reindex(sizes)
                bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, mean, width, yerr=std, capsize=4,
                              label=m, color=colors.get(m))
                for b, v, e in zip(bars, mean, std.fillna(0)):
                    ax.annotate(f"{v:.3f}" if "ssim" in metric or metric == "time_s" else f"{v:.2f}",
                                (b.get_x() + b.get_width() / 2, v + e), ha="center", va="bottom", fontsize=8,
                                xytext=(0, 2), textcoords="offset points")
            g_all = df.groupby(["method", "hole_target_pct"])[metric]
            ax.set_ylim(0, float((g_all.mean() + g_all.std().fillna(0)).max()) * 1.25)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s:g}% hole" for s in sizes])
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
        fig.suptitle(f"{' vs '.join(methods)}, mean ± std over {df['image'].nunique()} images")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame, out_dir: str) -> dict:
    agg = {m: ["mean", "std", "median"] for m in METRICS}
    overall = df.groupby("method").agg(agg)
    overall.columns = [f"{a}_{b}" for a, b in overall.columns]
    overall.insert(0, "n_tests", df.groupby("method").size())
    overall.round(4).to_csv(os.path.join(out_dir, "summary_overall.csv"))

    by_size = df.groupby(["hole_target_pct", "method"]).agg(agg)
    by_size.columns = [f"{a}_{b}" for a, b in by_size.columns]
    by_size.insert(0, "hole_actual_pct_mean", df.groupby(["hole_target_pct", "method"])["hole_actual_pct"].mean())
    by_size.insert(0, "n_tests", df.groupby(["hole_target_pct", "method"]).size())
    by_size.round(4).to_csv(os.path.join(out_dir, "summary_by_hole_size.csv"))

    wins = None
    if {"Telea", "LaMa"} <= set(df["method"]):
        key = ["image", "hole_target_pct"]
        t = df[df["method"] == "Telea"].set_index(key)
        l = df[df["method"] == "LaMa"].set_index(key)
        both = t.join(l, lsuffix="_telea", rsuffix="_lama", how="inner")
        rows = []
        for size, g in list(both.groupby(level="hole_target_pct")) + [("all", both)]:
            rows.append({
                "hole_target_pct": size,
                "n_tests": len(g),
                "lama_wins_psnr_hole": int((g["psnr_hole_lama"] > g["psnr_hole_telea"]).sum()),
                "lama_wins_ssim_hole": int((g["ssim_hole_lama"] > g["ssim_hole_telea"]).sum()),
                "mean_psnr_hole_gain_db": round(float((g["psnr_hole_lama"] - g["psnr_hole_telea"]).mean()), 4),
                "mean_ssim_hole_gain": round(float((g["ssim_hole_lama"] - g["ssim_hole_telea"]).mean()), 4),
                "mean_time_ratio_lama_over_telea": round(float((g["time_s_lama"] / g["time_s_telea"]).mean()), 2),
            })
        wins = pd.DataFrame(rows)
        wins.to_csv(os.path.join(out_dir, "wins_by_hole_size.csv"), index=False)
    return {"overall": overall, "by_size": by_size, "wins": wins}


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Benchmark OpenCV Telea vs LaMa inpainting against ground truth.")
    p.add_argument("--input", default=os.path.join("samples", "benchmark"), help="folder with ground-truth photos")
    p.add_argument("--out", default=os.path.join("outputs", "benchmark"), help="output folder")
    p.add_argument("--max-dim", type=int, default=512, help="resize so the long side is at most this (px)")
    p.add_argument("--coverages", default="5,15,30", help="hole sizes in percent, comma separated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--telea-radius", type=float, default=3.0)
    p.add_argument("--no-lama", action="store_true", help="run Telea only")
    p.add_argument("--limit", type=int, default=0, help="use only the first N photos (0 = all)")
    args = p.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    for name in ("InpaintingTelea", "InpaintingLaMa"):
        logging.getLogger(name).setLevel(logging.WARNING)

    paths = sorted(sum([glob.glob(os.path.join(args.input, e)) for e in ("*.png", "*.jpg", "*.jpeg")], []))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"No photos found in {args.input}. Run get_photos.py first.")
    coverages = [float(c) for c in args.coverages.split(",") if c.strip()]

    out_dir = args.out
    ex_dir = os.path.join(out_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)

    methods = {"Telea": lambda img, m: inpaint_telea(img, m, radius=args.telea_radius)}
    if not args.no_lama:
        import torch
        from utils.lama import inpaint_lama, get_lama_model
        logging.getLogger("InpaintingLaMa").setLevel(logging.WARNING)
        print("[*] Loading LaMa and warming up on a dummy image ...")
        t0 = time.perf_counter()
        get_lama_model()
        dummy = np.full((args.max_dim, args.max_dim, 3), 128, np.uint8)
        dmask = np.zeros(dummy.shape[:2], np.uint8)
        dmask[100:200, 100:200] = 255
        for _ in range(2):
            inpaint_lama(dummy, dmask)
        print(f"    model load + warm-up: {time.perf_counter() - t0:.1f} s (not counted), torch threads={torch.get_num_threads()}")
        methods["LaMa"] = inpaint_lama

    n_total = len(paths) * len(coverages)
    print(f"[*] {len(paths)} photos x {len(coverages)} hole sizes = {n_total} tests, methods: {', '.join(methods)}")
    print(f"[*] max-dim={args.max_dim}, seed={args.seed}, CPU={platform.processor()}")

    rows = []
    t_start = time.perf_counter()
    done = 0
    for img_idx, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        img = load_image(path, args.max_dim)
        h, w = img.shape[:2]
        for cov in coverages:
            rng = np.random.default_rng([args.seed, img_idx, int(round(cov * 100))])
            mask = make_mask(h, w, cov / 100.0, rng)
            actual = 100.0 * np.count_nonzero(mask) / mask.size
            corrupted = img.copy()
            corrupted[mask > 0] = 0

            results, test_rows = {}, {}
            for mname, fn in methods.items():
                t0 = time.perf_counter()
                res = fn(corrupted, mask)
                dt = time.perf_counter() - t0
                met = compute_metrics(img, res, mask)
                row = {"image": name, "width": w, "height": h, "hole_target_pct": cov,
                       "hole_actual_pct": round(actual, 3), "method": mname, **met, "time_s": dt}
                rows.append(row)
                results[mname], test_rows[mname] = res, row

            save_example(os.path.join(ex_dir, f"{name}_hole{cov:g}.png"), img, mask, results, test_rows)
            done += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (n_total - done)
            msg = "  ".join(f"{m}: {r['psnr_hole']:.2f} dB {r['ssim_hole']:.3f} {r['time_s']:.2f}s" for m, r in test_rows.items())
            print(f"[{done:3d}/{n_total}] {name} hole {cov:g}% (actual {actual:.1f}%)  {msg}  | ETA {eta / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results_all.csv"), index=False, float_format="%.5f")
    s = summarise(df, out_dir)
    save_charts(df, out_dir)

    print("\n=== Mean per method ===")
    print(df.groupby("method")[METRICS].mean().round(4).to_string())
    print("\n=== Mean per hole size ===")
    print(df.groupby(["hole_target_pct", "method"])[METRICS].mean().round(4).to_string())
    if s["wins"] is not None:
        print("\n=== LaMa vs Telea (per test) ===")
        print(s["wins"].to_string(index=False))
        allrow = s["wins"].iloc[-1]
        print(f"\nLaMa beat Telea in {allrow['lama_wins_psnr_hole']} of {allrow['n_tests']} tests (hole PSNR)")
        print(f"LaMa beat Telea in {allrow['lama_wins_ssim_hole']} of {allrow['n_tests']} tests (hole SSIM)")
    print(f"\nTotal wall time: {(time.perf_counter() - t_start) / 60:.1f} min. Results in {out_dir}")


if __name__ == "__main__":
    main()
