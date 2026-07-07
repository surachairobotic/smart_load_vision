import shutil
from pathlib import Path

video_dir = Path(r"D:\SASTech\06-Video AI\Gantry 15 AI Video\Videos")
process_dir = video_dir / "process"
non_process_dir = video_dir / "non-process"
partial_dir = video_dir / "partial_incomplete"

process_dir.mkdir(exist_ok=True)
non_process_dir.mkdir(exist_ok=True)
partial_dir.mkdir(exist_ok=True)

summary = {"process": 0, "non_process": 0, "partial": 0, "skipped": 0}

for f in sorted(video_dir.iterdir()):
    if not f.is_file():
        continue
    name = f.name

    if f.suffix == ".part":
        dest = partial_dir / name
        shutil.move(str(f), str(dest))
        summary["partial"] += 1
        print(f"[PARTIAL] {name}")
        continue

    if "detection" in name.lower():
        dest = process_dir / name
        shutil.move(str(f), str(dest))
        summary["process"] += 1
        print(f"[PROCESS] {name}")
    else:
        dest = non_process_dir / name
        shutil.move(str(f), str(dest))
        summary["non_process"] += 1
        print(f"[NON-PROCESS] {name}")

print("\n=== Summary ===")
print(f"Process (detection):       {summary['process']}")
print(f"Non-process (raw):         {summary['non_process']}")
print(f"Partial/incomplete (.part): {summary['partial']}")
print(f"Total: {sum(summary.values())}")
