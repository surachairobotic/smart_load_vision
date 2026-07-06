#!/usr/bin/env python3
"""
Run full video diagnosis pipeline.

Usage:
    python scripts/run_diagnosis.py --video-dir videos --output-dir output
    python scripts/run_diagnosis.py --video-dir videos --extract-frames --num-frames 20
    python scripts/run_diagnosis.py --video-dir videos --extract-keyframes --threshold 25.0
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnose import diagnose_all
from src.extract_frames import extract_frames_uniform, extract_keyframes


def main():
    parser = argparse.ArgumentParser(description="Smart Load Vision - Video Diagnosis")
    parser.add_argument("--video-dir", default="videos", help="Directory with video files")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--sample", action="store_true", default=True, help="Sample frame metadata")
    parser.add_argument("--no-sample", action="store_false", dest="sample")
    parser.add_argument("--extract-frames", action="store_true", help="Extract uniform frames")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract")
    parser.add_argument("--extract-keyframes", action="store_true", help="Extract keyframes by scene change")
    parser.add_argument("--threshold", type=float, default=30.0, help="Keyframe detection threshold")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_diagnosis")

    logger.info("Scanning videos in: %s", args.video_dir)

    results = diagnose_all(args.video_dir, args.output_dir, sample=args.sample)
    logger.info("Diagnosis complete: %d videos processed", len(results))

    if args.extract_frames:
        from src.utils import find_video_files, ensure_dir
        frames_dir = ensure_dir(str(Path(args.output_dir) / "frames"))
        for vp in find_video_files(args.video_dir):
            logger.info("Extracting %d frames from %s", args.num_frames, vp.name)
            extract_frames_uniform(str(vp), str(frames_dir), num_frames=args.num_frames)

    if args.extract_keyframes:
        from src.utils import find_video_files, ensure_dir
        kf_dir = ensure_dir(str(Path(args.output_dir) / "keyframes"))
        for vp in find_video_files(args.video_dir):
            logger.info("Extracting keyframes from %s (threshold=%.1f)", vp.name, args.threshold)
            extract_keyframes(str(vp), str(kf_dir), threshold=args.threshold)


if __name__ == "__main__":
    main()
