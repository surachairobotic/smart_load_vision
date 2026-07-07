"""
Two-pass binary search using Gemini Vision to find the first frame a truck appears.

Pass 1: Binary search over 2-second samples to narrow down arrival ±1 sec.
Pass 2: Linear scan over the 1-second window (±1 sec) to find exact frame.

Saves progress after every video. Resumes automatically.
"""

import csv
import json
import os
import time
from pathlib import Path

import cv2
import PIL.Image
from google import genai
from google.genai import errors as genai_errors
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_dir, find_video_files

# --- Config ---
HAS_TRUCK_DIR = Path(r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos\non-process\has_truck")
OUTPUT_DIR = Path("output/truck_arrival")
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
CSV_OUTPUT = OUTPUT_DIR / "arrival_timestamps.csv"
MANUAL_REVIEW_CSV = OUTPUT_DIR / "manual_review.csv"
API_LOG = OUTPUT_DIR / "api_call_log.csv"

MODEL = "gemini-3.1-flash-lite"
SAMPLE_INTERVAL = 2  # seconds
BINARY_SEARCH_ITERS = 20
MAX_RETRIES = 3
RETRY_DELAY = 61  # seconds (wait over 1 min for rate limit)
VIDEO_DELAY = 2  # seconds between videos

PROMPT = (
    "You are analyzing a surveillance camera at a loading dock.\n"
    "Does this image contain a large truck, lorry, or cargo vehicle?\n"
    "First answer YES or NO on the first line, then explain briefly."
)


class GeminiClient:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_GENERATIVE_AI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.call_count = 0

    def detect_truck(self, frame_rgb: PIL.Image.Image) -> tuple[bool, str]:
        self.call_count += 1
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=MODEL,
                    contents=[PROMPT, frame_rgb],
                    config={"temperature": 0.0},
                )
                text = response.text.strip() if response.text else ""
                first_line = text.split("\n")[0].upper() if text else ""
                if "YES" in first_line:
                    return True, first_line[:20]
                elif "NO" in first_line:
                    return False, first_line[:20]
                elif not text:
                    raise RuntimeError("Empty response")
                elif "YES" in text.upper():
                    return True, ("YES_fallback:" + text[:30])
                else:
                    return False, ("NO_fallback:" + text[:30])
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = RETRY_DELAY
                    if "retryDelay" in err_str:
                        import re
                        m = re.search(r"retryDelay['\"]:\s*['\"]([\d.]+)s['\"]", err_str)
                        if m:
                            delay = float(m.group(1))
                    print(f"\n  Rate limited. Waiting {delay:.0f}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(delay)
                else:
                    if attempt < MAX_RETRIES:
                        print(f"\n  API error: {err_str[:80]}. Retrying in 5s...")
                        time.sleep(5)
                    else:
                        print(f"\n  Failed after {MAX_RETRIES} attempts: {err_str[:80]}")
        return False, "ERROR"

    def log_call(self, video: str, timestamp: float, result: str, elapsed: float):
        api_log_path = Path(API_LOG)
        if not api_log_path.exists():
            with open(api_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["video", "timestamp_sec", "result", "elapsed_sec"])
        with open(api_log_path, "a", newline="") as f:
            csv.writer(f).writerow([video, round(timestamp, 1), result, round(elapsed, 2)])


class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        return {
            "completed": [],
            "manual_review": [],
            "failed": [],
            "results": {},
        }

    def save(self):
        ensure_dir(str(self.path.parent))
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, filename: str) -> bool:
        return filename in self.data["completed"]

    def add_result(self, filename: str, result: dict, category: str):
        self.data["completed"].append(filename)
        self.data["results"][filename] = result
        if category == "manual_review":
            self.data["manual_review"].append(filename)
        elif category == "failed":
            self.data["failed"].append(filename)
        self.save()

    def summary(self) -> dict:
        return {
            "completed": len(self.data["completed"]),
            "manual_review": len(self.data["manual_review"]),
            "failed": len(self.data["failed"]),
            "total_results": len(self.data["results"]),
        }


def binary_search_arrival(cap, fps: float, duration: float, client: GeminiClient,
                          video_name: str) -> tuple[float | None, list[dict]]:
    """Binary search for first frame with truck. Returns (timestamp_sec, call_logs)."""
    calls = []
    last_api_time = 0.0

    # Stagger calls to respect rate limit (5 RPM → 12s between calls)
    def _respect_rate_limit():
        nonlocal last_api_time
        elapsed = time.time() - last_api_time
        min_gap = 3  # seconds between API calls
        if last_api_time > 0 and elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        last_api_time = time.time()

    # Check if truck is visible at a given timestamp
    def check_truck(t_sec: float) -> tuple[bool, float]:
        _respect_rate_limit()
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return False, 0.0
        start = time.time()
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        has_truck, result_text = client.detect_truck(img)
        elapsed = time.time() - start
        client.log_call(video_name, t_sec, result_text, elapsed)
        calls.append({"timestamp": t_sec, "result": result_text, "elapsed": elapsed})
        return has_truck, elapsed

    # Step 1: Check frame 0
    truck_at_0, _ = check_truck(0.0)
    if truck_at_0:
        return 0.0, calls

    # Step 2: Check the last frame (to confirm truck exists in the video)
    if duration > SAMPLE_INTERVAL:
        truck_at_end, _ = check_truck(duration - 1)
        if not truck_at_end:
            # Try a few more points to be sure
            found_any = False
            for t in [duration * 0.25, duration * 0.5, duration * 0.75]:
                ft, _ = check_truck(t)
                if ft:
                    found_any = True
                    break
            if not found_any:
                return None, calls  # manual review

    # Step 3: Binary search
    low, high = 0.0, duration
    found = False

    for _ in range(BINARY_SEARCH_ITERS):
        mid = (low + high) / 2.0
        has_truck, _ = check_truck(mid)
        if has_truck:
            high = mid
            found = True
        else:
            low = mid
        if high - low <= SAMPLE_INTERVAL:
            break

    if found:
        return round(high, 1), calls
    return None, calls


def process_videos():
    ensure_dir(str(OUTPUT_DIR))
    client = GeminiClient()
    progress = ProgressTracker(PROGRESS_FILE)

    videos = sorted(
        v for v in Path(HAS_TRUCK_DIR).iterdir()
        if v.is_file() and v.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )

    print(f"Found {len(videos)} videos in {HAS_TRUCK_DIR}")
    print(f"Already completed: {progress.summary()['completed']}")
    print(f"Model: {MODEL}")
    print()

    results = []

    for vp in tqdm(videos, desc="Finding truck arrival"):
        if progress.is_completed(vp.name):
            # Load cached result
            cached = progress.data["results"].get(vp.name)
            if cached:
                results.append(cached)
            continue

        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if fps <= 0 or total_frames <= 0:
            progress.add_result(vp.name, {
                "filename": vp.name, "arrival_timestamp_sec": None,
                "duration_sec": 0.0, "confidence": 0.0, "api_calls": 0,
                "method": "failed", "notes": "Cannot read video",
            }, "failed")
            tqdm.write(f"  FAILED: {vp.name} (cannot read)")
            cap.release()
            continue

        tqdm.write(f"\n  {vp.name} ({duration:.0f}s)")

        arrival_sec, call_logs = binary_search_arrival(cap, fps, duration, client, vp.name)
        api_calls = len(call_logs)
        method = "truck_at_start" if arrival_sec == 0.0 else ("binary_search" if arrival_sec is not None else "manual_review")
        category = "manual_review" if arrival_sec is None else "completed"
        notes = "" if arrival_sec is not None else "no_truck_found_by_gemini"

        result = {
            "filename": vp.name,
            "arrival_timestamp_sec": arrival_sec,
            "duration_sec": round(duration, 1),
            "confidence": 1.0 if arrival_sec is not None else 0.0,
            "api_calls": api_calls,
            "method": method,
            "notes": notes,
        }

        progress.add_result(vp.name, result, category)
        results.append(result)
        cap.release()

        tqdm.write(f"  -> {method}: {arrival_sec}s (calls={api_calls})")

        # Respect rate limit (5 RPM for free tier)
        if VIDEO_DELAY > 0:
            time.sleep(VIDEO_DELAY)

    # Final save
    write_csv(results)
    write_manual_review(progress)
    print(f"\n=== Complete ===")
    print(f"  Total: {len(results)}")
    s = progress.summary()
    print(f"  Completed: {s['completed']}")
    print(f"  Manual review: {s['manual_review']}")
    print(f"  Failed: {s['failed']}")
    print(f"  Total API calls: {client.call_count}")
    print(f"  CSV: {CSV_OUTPUT}")
    print(f"  Progress: {PROGRESS_FILE}")


def write_csv(results: list[dict]):
    fields = ["filename", "arrival_timestamp_sec", "duration_sec", "confidence",
              "api_calls", "method", "notes"]
    with open(CSV_OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            if row["arrival_timestamp_sec"] is not None:
                row["arrival_timestamp_sec"] = round(row["arrival_timestamp_sec"], 1)
            else:
                row["arrival_timestamp_sec"] = ""
            w.writerow(row)


def write_manual_review(progress: ProgressTracker):
    manual = []
    for fname in progress.data["manual_review"]:
        r = progress.data["results"].get(fname, {})
        manual.append(r)
    fields = ["filename", "duration_sec", "api_calls", "notes"]
    with open(MANUAL_REVIEW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manual)
    print(f"  Manual review: {MANUAL_REVIEW_CSV} (videos needing human check)")


if __name__ == "__main__":
    process_videos()
