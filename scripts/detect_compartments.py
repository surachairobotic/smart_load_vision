import os, cv2, PIL.Image, json, time, re, csv, sys
from collections import OrderedDict

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
API_LOG = os.path.join(OUTPUT_DIR, "api_call_log.csv")

MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview"]
MAX_VIDEOS = 80
FRAME_OFFSETS = [60, 90, 120, 180, 30]
CROP_HEIGHT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamps = OrderedDict()
with open(TIMESTAMPS_CSV, newline='') as f:
    for row in csv.DictReader(f):
        timestamps[row['filename']] = float(row['arrival_timestamp_sec'])

all_videos = sorted([f for f in os.listdir(HAS_TRUCK_DIR) if f.endswith(('.mp4', '.avi', '.mov')) and f in timestamps])
print(f"Total videos: {len(all_videos)}")

progress = {}
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["filename", "compartment_count", "confidence", "x_centers", "frame_offset_sec", "model", "notes"])

if not os.path.exists(API_LOG):
    with open(API_LOG, 'w', newline='') as f:
        csv.writer(f).writerow(["video_name", "offset_sec", "model", "response", "elapsed_s", "success"])

def log_api(video, offset, model, response_text, elapsed, success):
    with open(API_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([video, round(offset, 1), model, response_text[:300], round(elapsed, 1), int(success)])

def analyze_frame(video_path, seek_sec, model_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    seek_sec = min(seek_sec, max(0, duration - 3))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return -1, [], f"frame_error"
    h, w = frame.shape[:2]
    crop_h = min(CROP_HEIGHT, h)
    crop = frame[0:crop_h, 0:w]
    ch, cw = crop.shape[:2]
    img = PIL.Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    prompt = f"""Image: {cw}x{ch}. Top portion of a tanker truck at a fuel loading dock.
Look at the very top edge of the truck's cylindrical tank. The COMPARTMENT HEADS (circular/oval manhole covers, domed protrusions) are visible along the top ridge.
Carefully count how many compartment heads you can see, from left to right. Typical tanker trucks have 3-8 compartments.
Return ONLY valid JSON:
{{"count": <int>, "compartment_centers_x": [<int>, <int>, ...], "notes": "<str>"}}
The centers_x array must have exactly 'count' entries, sorted left to right."""
    start = time.time()
    try:
        response = client.models.generate_content(model=model_name, contents=[prompt, img], config={"temperature": 0.0})
        text = response.text.strip() if response.text else ""
        elapsed = time.time() - start
    except Exception as e:
        elapsed = time.time() - start
        text = f"API_ERROR: {e}"
    data = None
    count = 0
    centers = []
    notes = ""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            count = data.get('count', 0)
            centers = data.get('compartment_centers_x', [])
            notes = data.get('notes', '')
            if not isinstance(count, int) or count < 0:
                count = 0
            if not isinstance(centers, list) or len(centers) != count:
                centers = []
        except:
            pass
    return count, centers, notes, text, elapsed

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)  # force flush
processed = 0
queue = [v for v in all_videos if v not in progress]
print(f"Queued: {len(queue)} videos, Models: {MODELS}")

all_exhausted = False

for video_name in queue:
    if processed >= MAX_VIDEOS:
        print(f"\nReached limit of {MAX_VIDEOS}. Resume tomorrow.")
        break

    arrival = timestamps.get(video_name, 0)
    count_final = 0
    centers_final = []
    offset_final = 0
    model_final = "none"
    notes_final = ""
    found = False
    models_429 = 0

    for offset in FRAME_OFFSETS:
        if found:
            break
        if all_exhausted:
            break
        seek_sec = arrival + offset
        vp = os.path.join(HAS_TRUCK_DIR, video_name)
        cap_tmp = cv2.VideoCapture(vp)
        duration = float(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) / cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()
        if seek_sec >= duration - 3:
            continue

        for model_name in MODELS:
            text, elapsed = "", 0
            try:
                print(f"  trying {video_name} @ {seek_sec}s with {model_name}...", end="", flush=True)
                count, centers, notes, text, elapsed = analyze_frame(vp, seek_sec, model_name)
                print(f" done", flush=True)
                if text.startswith("API_ERROR: 429"):
                    models_429 += 1
                    if models_429 >= len(MODELS) * len(FRAME_OFFSETS):
                        all_exhausted = True
                        print(f"  ALL MODELS EXHAUSTED. Stopping.")
                        break
                    print(f"  {model_name} exhausted on {video_name}")
                    continue
                if text.startswith("API_ERROR:"):
                    log_api(video_name, seek_sec, model_name, text, elapsed, 0)
                    continue
                log_api(video_name, seek_sec, model_name, text, elapsed, 1)
                if count > 0:
                    count_final = count
                    centers_final = centers
                    offset_final = seek_sec
                    model_final = model_name
                    notes_final = notes
                    found = True
                    print(f"  {video_name} -> count={count} @ {seek_sec}s model={model_name} centers={centers}")
                    break
                else:
                    print(f"  {video_name} @ {seek_sec}s {model_name}: count=0 ({notes[:60]})")
            except Exception as e:
                print(f"  {video_name} @ {seek_sec}s {model_name}: ERROR {str(e)[:80]}")

    if all_exhausted:
        print(f"  STOPPED - all API models exhausted. Processed {processed} this session.")
        break

    if not found:
        print(f"  {video_name} -> NO compartments detected")

    progress[video_name] = {"count": count_final, "centers": centers_final, "offset_sec": offset_final, "model": model_final, "notes": notes_final}
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([video_name, count_final, "1.0" if count_final > 0 else "0.0",
                                ";".join(str(c) for c in centers_final), offset_final, model_final, notes_final])
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    processed += 1

print(f"\nProcessed: {processed}")
rows = list(csv.DictReader(open(OUTPUT_CSV)))
counts = {}
for r in rows:
    c = int(r['compartment_count'])
    counts[c] = counts.get(c, 0) + 1
print(f"Distribution:")
for c in sorted(counts.keys()):
    print(f"  {c}: {counts[c]}")
print(f"  Total: {sum(counts.values())}")
