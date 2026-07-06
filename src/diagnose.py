import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4))

    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "filename": Path(video_path).name,
        "path": str(video_path),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "duration_hms": _format_duration(duration_sec),
        "codec": codec_str.strip(),
        "file_size_bytes": Path(video_path).stat().st_size,
        "file_size_mb": round(Path(video_path).stat().st_size / (1024 * 1024), 2),
    }


def sample_frames(video_path: str, num_samples: int = 5) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            mean_brightness = float(np.mean(frame))
            frames.append({
                "frame_index": int(idx),
                "timestamp_sec": round(int(idx) / cap.get(cv2.CAP_PROP_FPS), 3) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                "shape": f"{h}x{w}",
                "mean_brightness": round(mean_brightness, 2),
            })

    cap.release()
    return frames


def diagnose_video(video_path: str, sample: bool = True) -> dict:
    result = extract_video_metadata(video_path)
    if sample:
        result["frame_samples"] = sample_frames(video_path)
    return result


def diagnose_all(video_dir: str, output_dir: str = "output", sample: bool = True) -> list[dict]:
    from .utils import find_video_files, ensure_dir

    videos = find_video_files(video_dir)
    if not videos:
        logger.warning("No video files found in %s", video_dir)
        return []

    ensure_dir(output_dir)
    results = []

    for vp in tqdm(videos, desc="Diagnosing videos"):
        try:
            info = diagnose_video(str(vp), sample=sample)
            results.append(info)
        except Exception as e:
            logger.error("Failed to process %s: %s", vp.name, e)
            results.append({"filename": vp.name, "error": str(e)})

    report_path = Path(output_dir) / "diagnosis_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    summary_path = Path(output_dir) / "diagnosis_summary.csv"
    _write_csv(results, summary_path)

    return results


def _format_duration(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _write_csv(results: list[dict], path: Path):
    import csv
    keys = ["filename", "resolution", "fps", "total_frames", "duration_sec",
            "duration_hms", "codec", "file_size_mb", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
