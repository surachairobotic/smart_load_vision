import json
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import find_video_files, ensure_dir

NON_PROCESS_DIR = r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos\non-process"
OUTPUT_DIR = "output/threshold_diagnosis"
SAMPLE_INTERVAL_SEC = 15
MOTION_THRESHOLD = 30
INITIAL_BG_SAMPLES = 5
RUNNING_BG_WINDOW = 10
MAX_SAMPLES_PER_VIDEO = 1000


def analyze_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = int(fps * SAMPLE_INTERVAL_SEC)
    frame_skip = max(frame_skip, 1)
    video_name = Path(video_path).name
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = width * height

    samples = []
    frame_idx = 0
    initial_bg_frames = []
    running_bg_frames = []
    prev_gray = None

    sample_count = 0
    while sample_count < MAX_SAMPLES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Collect initial background ----
        if sample_count < INITIAL_BG_SAMPLES:
            initial_bg_frames.append(gray)

        # ---- Frame-to-frame motion ----
        motion_pct = 0.0
        max_contour_pct = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            motion_pct = float(cv2.countNonZero(thresh)) / total_pixels * 100.0

            # Largest contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_area = max(cv2.contourArea(c) for c in contours)
                max_contour_pct = max_area / total_pixels * 100.0
        prev_gray = gray

        # ---- Scene change from initial background ----
        scene_diff_pct = 0.0
        if len(initial_bg_frames) >= INITIAL_BG_SAMPLES:
            initial_bg = np.median(initial_bg_frames[:INITIAL_BG_SAMPLES], axis=0).astype(np.uint8)
            sdiff = cv2.absdiff(gray, initial_bg)
            _, sthresh = cv2.threshold(sdiff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            scene_diff_pct = float(cv2.countNonZero(sthresh)) / total_pixels * 100.0

        # ---- Running background diff ----
        running_bg = None
        run_diff_pct = 0.0
        if len(running_bg_frames) >= RUNNING_BG_WINDOW:
            running_bg = np.median(running_bg_frames[-RUNNING_BG_WINDOW:], axis=0).astype(np.uint8)
            rdiff = cv2.absdiff(gray, running_bg)
            _, rthresh = cv2.threshold(rdiff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            run_diff_pct = float(cv2.countNonZero(rthresh)) / total_pixels * 100.0
        running_bg_frames.append(gray)

        # Keep running bg window bounded
        if len(running_bg_frames) > RUNNING_BG_WINDOW * 2:
            running_bg_frames = running_bg_frames[-RUNNING_BG_WINDOW:]

        brightness = float(np.mean(gray))

        timestamp_sec = frame_idx / fps

        samples.append({
            "timestamp_sec": round(timestamp_sec, 1),
            "motion_pct": round(motion_pct, 4),
            "max_contour_pct": round(max_contour_pct, 4),
            "scene_diff_pct": round(scene_diff_pct, 4),
            "run_diff_pct": round(run_diff_pct, 4),
            "brightness": round(brightness, 2),
        })

        frame_idx += frame_skip
        sample_count += 1

    cap.release()

    if not samples:
        return None

    # Per-video summary stats
    motion_vals = np.array([s["motion_pct"] for s in samples])
    scene_vals = np.array([s["scene_diff_pct"] for s in samples])
    run_vals = np.array([s["run_diff_pct"] for s in samples])
    contour_vals = np.array([s["max_contour_pct"] for s in samples])
    bright_vals = np.array([s["brightness"] for s in samples])

    def percentiles(arr):
        if len(arr) == 0:
            return {}
        return {
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
        }

    stats = {
        "filename": video_name,
        "width": width,
        "height": height,
        "total_samples": len(samples),
        "duration_sec": round(samples[-1]["timestamp_sec"], 1),
        "motion": percentiles(motion_vals),
        "scene_diff": percentiles(scene_vals),
        "run_diff": percentiles(run_vals),
        "max_contour": percentiles(contour_vals),
        "brightness": percentiles(bright_vals),
        "samples": samples,
    }
    return stats


def plot_histograms(all_stats: list[dict], output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Aggregate across all samples
    all_motion = []
    all_scene = []
    all_run = []
    all_contour = []
    all_brightness = []

    for vs in all_stats:
        for s in vs["samples"]:
            all_motion.append(s["motion_pct"])
            all_scene.append(s["scene_diff_pct"])
            all_run.append(s["run_diff_pct"])
            all_contour.append(s["max_contour_pct"])
            all_brightness.append(s["brightness"])

    features = [
        ("motion_pct", all_motion, "Frame-to-frame Motion (%)", 0, 0.1, 50),
        ("scene_diff_pct", all_scene, "Scene Change from Initial BG (%)", 0, 0.1, 50),
        ("run_diff_pct", all_run, "Scene Change from Running BG (%)", 0, 0.1, 50),
        ("max_contour_pct", all_contour, "Max Contour Area (%)", 0, 0.1, 50),
        ("brightness", all_brightness, "Mean Brightness", 0, 1, 50),
    ]

    for name, data, xlabel, vmin, clip_min, nbins in features:
        arr = np.array(data)
        arr = arr[arr >= vmin]
        if len(arr) == 0:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Full histogram (clipped at 95th percentile for visibility)
        clip_max = float(np.percentile(arr, 95)) if len(arr) > 100 else float(arr.max())
        clipped = arr[arr <= clip_max]
        ax1.hist(clipped, bins=nbins, edgecolor="black", alpha=0.7)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Count")
        ax1.set_title(f"{name} (clipped at p95={clip_max:.2f})")
        ax1.grid(True, alpha=0.3)

        # Log-scale histogram
        ax2.hist(clipped, bins=nbins, edgecolor="black", alpha=0.7)
        ax2.set_yscale("log")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Count (log)")
        ax2.set_title(f"{name} (log scale)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"histogram_{name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Per-video max motion sorted bar chart
    videos_sorted = sorted(all_stats, key=lambda v: v["motion"]["max"], reverse=True)
    names = [v["filename"] for v in videos_sorted[:60]]
    max_motion = [v["motion"]["max"] for v in videos_sorted[:60]]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(len(names)), max_motion)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("Max Motion (%)")
    ax.set_title("Top 60 Videos by Max Motion")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = output_dir / "top60_max_motion.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_csv_summary(all_stats: list[dict], path: Path):
    fields = [
        "filename", "width", "height", "total_samples", "duration_sec",
        "motion_min", "motion_max", "motion_mean", "motion_median", "motion_p90", "motion_p95", "motion_p99",
        "scene_diff_min", "scene_diff_max", "scene_diff_mean", "scene_diff_median", "scene_diff_p90", "scene_diff_p95", "scene_diff_p99",
        "run_diff_min", "run_diff_max", "run_diff_mean", "run_diff_median", "run_diff_p90", "run_diff_p95", "run_diff_p99",
        "max_contour_min", "max_contour_max", "max_contour_mean", "max_contour_median", "max_contour_p90", "max_contour_p95", "max_contour_p99",
        "brightness_mean",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for vs in all_stats:
            row = [
                vs["filename"], vs["width"], vs["height"], vs["total_samples"], vs["duration_sec"],
                vs["motion"]["min"], vs["motion"]["max"], vs["motion"]["mean"], vs["motion"]["median"],
                vs["motion"]["p90"], vs["motion"]["p95"], vs["motion"]["p99"],
                vs["scene_diff"]["min"], vs["scene_diff"]["max"], vs["scene_diff"]["mean"], vs["scene_diff"]["median"],
                vs["scene_diff"]["p90"], vs["scene_diff"]["p95"], vs["scene_diff"]["p99"],
                vs["run_diff"]["min"], vs["run_diff"]["max"], vs["run_diff"]["mean"], vs["run_diff"]["median"],
                vs["run_diff"]["p90"], vs["run_diff"]["p95"], vs["run_diff"]["p99"],
                vs["max_contour"]["min"], vs["max_contour"]["max"], vs["max_contour"]["mean"], vs["max_contour"]["median"],
                vs["max_contour"]["p90"], vs["max_contour"]["p95"], vs["max_contour"]["p99"],
                vs["brightness"]["mean"],
            ]
            w.writerow(row)


def main():
    out_dir = ensure_dir(OUTPUT_DIR)
    videos = find_video_files(NON_PROCESS_DIR)
    print(f"Found {len(videos)} videos in {NON_PROCESS_DIR}")

    all_stats = []
    failed = []

    for vp in tqdm(videos, desc="Analyzing videos"):
        try:
            stats = analyze_video(str(vp))
            if stats is None:
                failed.append(vp.name)
                continue
            all_stats.append(stats)

            # Save per-video JSON
            json_path = out_dir / f"{vp.stem}_stats.json"
            with open(json_path, "w") as f:
                # Exclude full sample array from per-video json to save space
                export = {k: v for k, v in stats.items() if k != "samples"}
                export["sample_count"] = len(stats["samples"])
                json.dump(export, f, indent=2)
        except Exception as e:
            failed.append(vp.name)
            tqdm.write(f"  ERROR: {vp.name}: {e}")

    print(f"\nProcessed: {len(all_stats)}, Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")

    # Write summary CSV
    csv_path = out_dir / "diagnosis_summary.csv"
    write_csv_summary(all_stats, csv_path)
    print(f"Summary CSV: {csv_path}")

    # Write aggregated JSON with all stats (excluding per-sample arrays)
    agg_path = out_dir / "diagnosis_aggregated.json"
    agg = []
    for vs in all_stats:
        entry = {k: v for k, v in vs.items() if k != "samples"}
        entry["sample_count"] = len(vs["samples"])
        agg.append(entry)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregated JSON: {agg_path}")

    # Write full sample data for threshold analysis
    sample_csv_path = out_dir / "all_samples.csv"
    with open(sample_csv_path, "w", newline="") as f:
        fields = ["filename", "timestamp_sec", "motion_pct", "max_contour_pct", "scene_diff_pct", "run_diff_pct", "brightness"]
        w = csv.writer(f)
        w.writerow(fields)
        for vs in all_stats:
            for s in vs["samples"]:
                w.writerow([vs["filename"], s["timestamp_sec"], s["motion_pct"], s["max_contour_pct"], s["scene_diff_pct"], s["run_diff_pct"], s["brightness"]])
    print(f"All samples CSV: {sample_csv_path}")

    # Plot histograms
    print("Generating plots...")
    plot_histograms(all_stats, out_dir)

    # Print suggested thresholds based on aggregated data
    print("\n=== Suggested Threshold Range (from aggregated data) ===")
    all_motion = np.array([s["motion_pct"] for vs in all_stats for s in vs["samples"]])
    all_scene = np.array([s["scene_diff_pct"] for vs in all_stats for s in vs["samples"]])
    all_contour = np.array([s["max_contour_pct"] for vs in all_stats for s in vs["samples"]])

    for name, arr in [("motion_pct", all_motion), ("scene_diff_pct", all_scene), ("max_contour_pct", all_contour)]:
        nonzero = arr[arr > 0.5]
        if len(nonzero) > 0:
            print(f"  {name}: p50={np.percentile(arr, 50):.2f}, p85={np.percentile(arr, 85):.2f}, p90={np.percentile(arr, 90):.2f}, p95={np.percentile(arr, 95):.2f}, p99={np.percentile(arr, 99):.2f}")
        else:
            print(f"  {name}: mostly zero")


if __name__ == "__main__":
    main()
