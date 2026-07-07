import os, cv2, PIL.Image, json, time, re, csv, sys, math

API_KEY = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
if not API_KEY:
    print("FATAL: GOOGLE_GENERATIVE_AI_API_KEY not set")
    sys.exit(1)

from google import genai
client = genai.Client(api_key=API_KEY)

HAS_TRUCK_DIR = r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos\non-process\has_truck"
TIMESTAMPS_CSV = r"D:\SASTech\smart_load_vision\output\truck_arrival\arrival_timestamps.csv"
OUTPUT_DIR = r"D:\SASTech\smart_load_vision\output\compartment_detection"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "compartment_counts.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
API_LOG = os.path.join(OUTPUT_DIR, "api_call_log_retry.csv")
MODEL = "gemini-3.1-flash-lite"
MAX_VIDEOS = 20
MAX_API_CALLS_PER_VIDEO = 7
CROP_HEIGHT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamps = {}
with open(TIMESTAMPS_CSV, newline='') as f:
    for row in csv.DictReader(f):
        timestamps[row['filename']] = float(row['arrival_timestamp_sec'])

if not os.path.exists(API_LOG):
    with open(API_LOG, 'w', newline='') as f:
        csv.writer(f).writerow(["video_name", "seek_sec", "response", "elapsed_s", "count"])

def log_api(video, seek, text, elapsed, count):
    with open(API_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([video, round(seek, 1), text[:200], round(elapsed, 1), count])

def count_compartments(video_path, seek_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    seek_sec = min(seek_sec, max(0, duration - 3))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return -1, [], "frame_error"
    h, w = frame.shape[:2]
    crop_h = min(CROP_HEIGHT, h)
    crop = frame[0:crop_h, 0:w]
    ch, cw = crop.shape[:2]
    img = PIL.Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    prompt = f"""Image: {cw}x{ch}. Top portion of a tanker truck at a fuel loading dock.

Look at the very top edge of the truck's cylindrical tank. The COMPARTMENT HEADS (circular/oval manhole covers, also called domed protrusions) are visible along the top ridge.

Carefully count how many compartment heads you can see, from left to right. Typical tanker trucks have 3-8 compartments.

Return ONLY valid JSON:
{{"count": <int>, "compartment_centers_x": [<int>, <int>, ...], "notes": "<str>"}}
The centers_x array must have exactly 'count' entries."""
    start = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL, contents=[prompt, img], config={"temperature": 0.0})
        text = response.text.strip() if response.text else ""
        elapsed = time.time() - start
    except Exception as e:
        text = f"API_ERROR: {e}"
        elapsed = time.time() - start
    data = None
    count = 0
    centers = []
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            count = data.get('count', 0)
            centers = data.get('compartment_centers_x', [])
            if not isinstance(count, int) or count < 0:
                count = 0
            if len(centers) != count:
                centers = []
        except:
            pass
    return count, centers, text, elapsed

def update_csv(video, count, centers, offset, method, notes):
    rows = []
    updated = False
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['filename'] == video:
                    row['compartment_count'] = str(count)
                    row['confidence'] = "1.0" if count > 0 else "0.0"
                    row['x_centers'] = ";".join(str(c) for c in centers)
                    row['frame_offset_sec'] = str(offset)
                    row['method'] = method
                    row['notes'] = notes
                    updated = True
                rows.append(row)
    else:
        fieldnames = ["filename", "compartment_count", "confidence", "x_centers", "frame_offset_sec", "method", "notes"]
    if not updated:
        rows.append({"filename": video, "compartment_count": str(count), "confidence": "1.0" if count > 0 else "0.0",
                      "x_centers": ";".join(str(c) for c in centers), "frame_offset_sec": str(offset),
                      "method": method, "notes": notes})
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def update_progress(video, count, centers, offset, method, notes):
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    progress[video] = {"count": count, "centers": centers, "offset_sec": offset, "method": method, "notes": notes}
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

# Load videos that need retry (count < 3)
retry_videos = []
with open(OUTPUT_CSV, newline='') as f:
    for row in csv.DictReader(f):
        c = int(row['compartment_count'])
        if c < 3:
            retry_videos.append(row['filename'])
print(f"Videos needing retry (count<3): {len(retry_videos)} -> {retry_videos}")

if not retry_videos:
    print("No retry needed!")
    sys.exit(0)

processed = 0
for video_name in retry_videos:
    if processed >= MAX_VIDEOS:
        print(f"\nReached daily limit of {MAX_VIDEOS}.")
        break
    arrival = timestamps.get(video_name, 0)
    vp = os.path.join(HAS_TRUCK_DIR, video_name)
    cap = cv2.VideoCapture(vp)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"\n=== {video_name} (arrival={arrival}s, duration={duration:.0f}s) ===")
    
    # Binary search: find frame with highest compartment count
    lo = max(0, arrival)
    hi = min(duration - 5, arrival + 360)
    best_count = 0
    best_centers = []
    best_seek = 0
    best_method = "binary_search"
    api_calls = 0
    tested = set()
    
    # Binary search iterations
    while lo < hi and api_calls < MAX_API_CALLS_PER_VIDEO:
        mid = (lo + hi) / 2
        # Round to nearest second to avoid too many unique points
        mid = round(mid)
        if mid in tested:
            break
        tested.add(mid)
        
        count, centers, text, elapsed = count_compartments(vp, mid)
        api_calls += 1
        log_api(video_name, mid, text, elapsed, count)
        print(f"  binary mid={mid}s: count={count}" + (f" centers={centers}" if count > 0 else ""))
        
        if count > best_count:
            best_count = count
            best_centers = centers
            best_seek = mid
        
        if count >= 5:
            print(f"  Found 5 compartments! Stopping.")
            break
        elif count >= 3:
            print(f"  Found {count} compartments (>=3). Stopping.")
            break
        elif count == 0:
            # No compartments visible - move to later part of video (truck should be parked)
            lo = mid + 10
        else:
            # count is 1-2, compartments partially visible - try both sides
            # Try a slightly different position
            lo = mid + 20
    
    # If binary search didn't find >=3, try some additional fixed offsets as fallback
    if best_count < 3 and api_calls < MAX_API_CALLS_PER_VIDEO:
        for fallback in [arrival + 90, arrival + 150, arrival + 240, duration / 2]:
            fallback = round(fallback)
            if 0 <= fallback < duration - 3 and fallback not in tested and api_calls < MAX_API_CALLS_PER_VIDEO:
                count, centers, text, elapsed = count_compartments(vp, fallback)
                api_calls += 1
                log_api(video_name, fallback, text, elapsed, count)
                print(f"  fallback={fallback}s: count={count}")
                if count > best_count:
                    best_count = count
                    best_centers = centers
                    best_seek = fallback
                if count >= 3:
                    break
    
    print(f"  FINAL: count={best_count} @ {best_seek}s")
    
    update_csv(video_name, best_count, best_centers, best_seek, best_method, f"retry_{api_calls}calls")
    update_progress(video_name, best_count, best_centers, best_seek, best_method, f"retry_{api_calls}calls")
    processed += 1

print(f"\nDone. Retried: {processed}")
