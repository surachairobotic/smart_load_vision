import json
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_dir

NON_PROCESS_DIR = Path(r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos\non-process")
DIAGNOSTIC_DIR = Path("output/threshold_diagnosis")
CONFIG_PATH = Path("output/threshold_diagnosis/classification_config.json")

DEFAULT_THRESHOLDS = {
    "motion_max": 12.0,
    "scene_diff_max": 40.0,
    "motion_mean": 3.0,
    "scene_diff_mean": 25.0,
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    save_config(DEFAULT_THRESHOLDS)
    return dict(DEFAULT_THRESHOLDS)


def save_config(cfg: dict):
    ensure_dir(str(CONFIG_PATH.parent))
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved: {CONFIG_PATH}")


def classify_video(stats: dict, thresholds: dict) -> tuple[str, float]:
    """
    Returns (category, confidence).
    Category: 'has_truck' or 'no_truck'
    Confidence: 0.0 (unsure) to 1.0 (very sure)
    """
    motion_max = stats["motion"]["max"]
    scene_max = stats["scene_diff"]["max"]
    motion_mean = stats["motion"]["mean"]
    scene_mean = stats["scene_diff"]["mean"]
    contour_max = stats["max_contour"]["max"]

    # Primary: large motion spike or major scene change
    if motion_max >= thresholds["motion_max"] or scene_max >= thresholds["scene_diff_max"]:
        confidence = max(
            0.5,
            min(1.0, motion_max / 60.0) if motion_max >= thresholds["motion_max"]
            else min(1.0, scene_max / 80.0)
        )
        return "has_truck", round(confidence, 3)

    # Secondary: consistent moderate motion + scene change
    if motion_mean >= thresholds["motion_mean"] and scene_mean >= thresholds["scene_diff_mean"]:
        confidence = round(min(1.0, (motion_mean + scene_mean) / 100.0), 3)
        return "has_truck", confidence

    return "no_truck", round(max(0.5, 1.0 - motion_max / thresholds["motion_max"]), 3)


def main():
    thresholds = load_config()
    print(f"Current thresholds: {json.dumps(thresholds, indent=2)}")

    if not NON_PROCESS_DIR.exists():
        print(f"ERROR: {NON_PROCESS_DIR} not found")
        sys.exit(1)

    # Collect all per-video stats from diagnostic output
    stats_files = sorted(DIAGNOSTIC_DIR.glob("*_stats.json"))
    if not stats_files:
        print("ERROR: No per-video stats found. Run detect_thresholds.py first.")
        sys.exit(1)

    print(f"Found {len(stats_files)} stat files")

    results = []
    has_truck_dir = NON_PROCESS_DIR.parent / "has_truck"
    no_truck_dir = NON_PROCESS_DIR.parent / "no_truck"
    ensure_dir(str(has_truck_dir))
    ensure_dir(str(no_truck_dir))

    for sf in stats_files:
        with open(sf) as f:
            stats = json.load(f)

        category, confidence = classify_video(stats, thresholds)
        src_path = NON_PROCESS_DIR / stats["filename"]

        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping")
            continue

        if category == "has_truck":
            dst = has_truck_dir / stats["filename"]
        else:
            dst = no_truck_dir / stats["filename"]

        shutil.move(str(src_path), str(dst))

        results.append({
            "filename": stats["filename"],
            "category": category,
            "confidence": confidence,
            "motion_max": stats["motion"]["max"],
            "motion_mean": stats["motion"]["mean"],
            "scene_diff_max": stats["scene_diff"]["max"],
            "scene_diff_mean": stats["scene_diff"]["mean"],
            "brightness_mean": stats["brightness"]["mean"],
        })

    # Summary
    truck_count = sum(1 for r in results if r["category"] == "has_truck")
    no_truck_count = sum(1 for r in results if r["category"] == "no_truck")

    print(f"\n=== Classification Complete ===")
    print(f"  has_truck: {truck_count} -> {has_truck_dir}")
    print(f"  no_truck: {no_truck_count} -> {no_truck_dir}")
    print(f"  Total: {len(results)}")

    # Write CSV report
    report_path = DIAGNOSTIC_DIR / "classification_report.csv"
    fields = ["filename", "category", "confidence", "motion_max", "motion_mean",
              "scene_diff_max", "scene_diff_mean", "brightness_mean"]
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Report: {report_path}")

    # Show threshold config for next use
    print(f"\nTo adjust thresholds, edit: {CONFIG_PATH}")
    print("Run again to reclassify (files will be re-moved)")


if __name__ == "__main__":
    main()
