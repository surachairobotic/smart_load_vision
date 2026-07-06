# Smart Load Vision - Video Diagnosis

Diagnose, inspect, and analyze loading dock videos using computer vision.

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

Place video files in `videos/`, then run:

```bash
# Quick diagnosis (metadata + frame samples)
python scripts/run_diagnosis.py

# Extract uniform frames from every video
python scripts/run_diagnosis.py --extract-frames --num-frames 20

# Extract keyframes based on scene changes
python scripts/run_diagnosis.py --extract-keyframes --threshold 25.0

# Output goes to output/diagnosis_report.json and output/diagnosis_summary.csv
```

## Project Structure

```
├── videos/              # Input videos (gitignored)
├── output/              # Diagnosis reports & extracted frames
├── src/
│   ├── diagnose.py      # Video metadata extraction
│   ├── extract_frames.py# Frame/keyframe extraction
│   └── utils.py         # File helpers
├── models/              # AI models (gitignored)
└── scripts/
    └── run_diagnosis.py # CLI entry point
```
