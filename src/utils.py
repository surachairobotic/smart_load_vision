import os
from pathlib import Path


def find_video_files(video_dir: str, extensions=None) -> list[Path]:
    if extensions is None:
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    found = []
    for f in sorted(video_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            found.append(f)
    return found


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(path: Path) -> str:
    return path.stem
