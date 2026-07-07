import os, cv2, json, csv, sys
from collections import OrderedDict

HAS_TRUCK_DIR = r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos\non-process\has_truck"
TIMESTAMPS_CSV = r"D:\SASTech\smart_load_vision\output\truck_arrival\arrival_timestamps.csv"
OUTPUT_DIR = r"D:\SASTech\smart_load_vision\output\compartment_detection"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "compartment_counts_manual.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_manual.json")
HIGHLIGHT_HEIGHT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamps = OrderedDict()
with open(TIMESTAMPS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        timestamps[row["filename"]] = float(row["arrival_timestamp_sec"])

all_videos = sorted([f for f in os.listdir(HAS_TRUCK_DIR) if f.endswith((".mp4", ".avi", ".mov")) and f in timestamps])
print(f"Total videos in has_truck/: {len(all_videos)}")

progress = {}
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

remaining = [v for v in all_videos if v not in progress]
print(f"Already labeled in this session: {len(progress)}")
print(f"Remaining: {len(remaining)}")

if not remaining:
    print("All done!")
    sys.exit(0)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["filename", "compartment_count", "notes"])

print()
print("=== MANUAL COMPARTMENT LABELING ===")
print(f"Total: {len(remaining)} videos, ~3 frames each")
print()
print("Controls per video:")
print("  1) 3 frames shown full-screen with top-200px highlighted")
print("  2) Press any key to cycle through frames")
print("  3) After frame 3, window stays open — enter count in terminal")
print()
print("  [0-8] = compartment count    [s] = skip video")
print("  [q]   = quit (progress saved)")
print()

for idx, video_name in enumerate(remaining, 1):
    vp = os.path.join(HAS_TRUCK_DIR, video_name)
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    arrival = timestamps.get(video_name, 0)

    safe_end = max(0, duration - 3)
    start = min(arrival, safe_end)
    range_sec = max(1, safe_end - start)
    offsets = [
        start + range_sec * 0.25,
        start + range_sec * 0.50,
        start + range_sec * 0.75,
    ]

    last_display = None
    for i, offset in enumerate(offsets):
        seek_frame = int(offset * fps)
        seek_frame = max(0, min(seek_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        display = frame.copy()
        cv2.rectangle(display, (0, 0), (w, min(HIGHLIGHT_HEIGHT, h)), (0, 255, 0), 2)
        lines = [
            f"{idx}/{len(remaining)}  {video_name}",
            f"Frame {i+1}/3  offset={offset:.0f}s  arrival={arrival:.0f}s",
            f"Look at TOP EDGE of tank  |  Press any key for next frame"
        ]
        for li, txt in enumerate(lines):
            cv2.putText(display, txt, (10, 25 + li*25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Manual Label - compartment head counting", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("\nQuit. Progress saved.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        last_display = display

    cap.release()

    cv2.imshow("Manual Label - enter count in terminal below (q=quit s=skip)", last_display)
    cv2.waitKey(1)

    while True:
        try:
            inp = input(f"  Count (0-8) [{idx}/{len(remaining)} {video_name}] (q=quit s=skip): ").strip()
            if inp.lower() == "q":
                print("Quit. Progress saved.")
                cv2.destroyAllWindows()
                sys.exit(0)
            if inp.lower() == "s":
                notes = "manual_skip"
                count = -1
                break
            count = int(inp)
            if 0 <= count <= 8:
                notes = "manual_label"
                break
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
    print(f"  -> {status}")
    print()

cv2.destroyAllWindows()
done = len([v for v, p in progress.items() if p.get("notes") == "manual_label"])
skipped = len([v for v, p in progress.items() if p.get("notes") == "manual_skip"])
print(f"\nSession complete!")
print(f"  Labeled: {done}")
print(f"  Skipped: {skipped}")
print(f"  Output: {OUTPUT_CSV}")
