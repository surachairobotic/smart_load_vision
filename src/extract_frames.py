import logging
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_frames_uniform(
    video_path: str,
    output_dir: str,
    num_frames: int = 10,
    format: str = "jpg",
) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    saved = []
    indices = set(
        int(round(i * (total - 1) / (num_frames - 1)))
        for i in range(num_frames)
    )

    for frame_idx in tqdm(range(total), desc=f"Extracting {stem}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            out_path = str(Path(output_dir) / f"{stem}_frame_{frame_idx:06d}.{format}")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)

    cap.release()
    return saved


def extract_keyframes(video_path: str, output_dir: str, threshold: float = 30.0) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    saved = []
    prev_gray = None
    frame_idx = 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc=f"Keyframes {stem}", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray).mean()
            if diff > threshold:
                out_path = str(Path(output_dir) / f"{stem}_keyframe_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved.append(out_path)
        prev_gray = gray
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return saved
