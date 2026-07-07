import os, cv2, json, csv, sys
from collections import OrderedDict

HAS_TRUCK_DIR = r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos\non-process\has_truck"
TIMESTAMPS_CSV = r"D:\SASTech\smart_load_vision\output\truck_arrival\arrival_timestamps.csv"
EXISTING_CSV = r"D:\SASTech\smart_load_vision\output\compartment_detection\compartment_counts.csv"
OUTPUT_DIR = r"D:\SASTech\smart_load_vision\output\compartment_detection"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "compartment_counts_manual.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_manual.json")
CROP_HEIGHT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamps = OrderedDict()
with open(TIMESTAMPS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        timestamps[row["filename"]] = float(row["arrival_timestamp_sec"])

existing = set()
if os.path.exists(EXISTING_CSV):
    with open(EXISTING_CSV, newline="") as f:
        for row in csv.DictReader(f):
            existing.add(row["filename"])

all_videos = sorted([f for f in os.listdir(HAS_TRUCK_DIR) if f.endswith((".mp4", ".avi", ".mov")) and f in timestamps])
remaining = [v for v in all_videos if v not in existing]
print(f"Total has_truck: {len(all_videos)}")
print(f"Already processed by AI: {len(existing)}")
print(f"Remaining for manual label: {len(remaining)}")

progress = {}
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    remaining = [v for v in remaining if v not in progress]
    print(f"After skipping already labeled: {len(remaining)}")

if not remaining:
    print("All done! No videos remaining.")
    sys.exit(0)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["filename", "compartment_count", "notes"])

print(f"\nManual label: {len(remaining)} videos")
print("Controls:")
print("  After viewing 3 frames, enter count (0-8) in terminal")
print("  Press 'q' at any frame to quit and save progress")
print("  Press 's' to skip current video")
print()

for idx, video_name in enumerate(remaining, 1):
    vp = os.path.join(HAS_TRUCK_DIR, video_name)
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    arrival = timestamps.get(video_name, 0)

    safe_end = max(0, duration - 3)
    start = min(arrival, safe_end)
    range_sec = safe_end - start
    offsets = [
        start + range_sec * 0.25,
        start + range_sec * 0.50,
        start + range_sec * 0.75,
    ]

    frames = []
    last_display = None
    for i, offset in enumerate(offsets):
        seek_frame = int(offset * fps)
        seek_frame = max(0, min(seek_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            continue
        h, w = frame.shape[:2]
        crop_h = min(CROP_HEIGHT, h)
        crop = frame[0:crop_h, 0:w]
        display = crop.copy()
        label = f"{idx}/{len(remaining)}  Frame {i+1}/3  offset={offset:.0f}s  key any -> next"
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Manual Label: press any key for next frame, then enter count in terminal", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("\nQuit requested. Progress saved.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        frames.append(crop)
        last_display = display

    cap.release()

    print(f"\nVideo {idx}/{len(remaining)}: {video_name}")
    print(f"  Arrival: {arrival:.0f}s  Duration: {duration:.0f}s")
    print(f"  Offsets: {[f'{o:.0f}s' for o in offsets]}")
    cv2.imshow("Manual Label: enter count in terminal below, q=quit s=skip", last_display)
    cv2.waitKey(1)
    while True:
        try:
            inp = input(f"  Count (0-8) [{video_name}] (q=quit s=skip): ").strip()
            if inp.lower() == "q":
                print("Quit. Progress saved.")
                cv2.destroyAllWindows()
                sys.exit(0)
            if inp.lower() == "s":
                print(f"  Skipped {video_name}")
                notes = "manual_skip"
                count = -1
                break
            count = int(inp)
            if 0 <= count <= 8:
                notes = "manual_label"
                break
            else:
                print("  Enter 0-8")
        except ValueError:
            print("  Invalid, enter 0-8")

    save_count = count if notes == "manual_label" else ""
    progress[video_name] = {"count": count if count >= 0 else None, "notes": notes, "offsets": [round(o, 1) for o in offsets]}
    with open(OUTPUT_CSV, "a", newline="") as f:
        csv.writer(f).writerow([video_name, save_count, notes])
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    status = f"count={count}" if notes == "manual_label" else "skipped"
    print(f"  {status}")

cv2.destroyAllWindows()
print(f"\nDone! Labeled {len(progress)} videos this session.")
