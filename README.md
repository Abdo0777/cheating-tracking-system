# 🎓 Exam Cheating Tracking & Alert System

A desktop AI application that monitors exam footage in real time, tracks
each student individually, flags suspicious behavior, and logs every
incident with a screenshot — all through a simple Tkinter interface.

Built with YOLO for detection, ByteTrack for persistent multi-object
tracking, and MediaPipe for head-pose estimation.

---

## Features

- 🎥 **Upload & process** any exam video through a desktop GUI
- 🧍 **Per-student tracking** — each student keeps a stable ID across frames, not just a frame-by-frame detection
- 🚨 **Cheating detection**
  - Phone held/near a student → flagged instantly
  - Sustained head-turn away from the desk → flagged after N consecutive frames
- 🟩🟥 **Live color-coded boxes** — green for normal, red the moment a student is flagged
- 📸 **Automatic screenshots** of every flagged moment, saved locally
- 🗄️ **Alert history** — timestamp, behavior, confidence score, and screenshot path, all queryable later
- 🔔 **Desktop notifications** the instant an incident is detected
- 🖼️ **Screenshot viewer** built into the app — no digging through folders

---

## Tech Stack

| Layer | Tool |
|---|---|
| Detection | YOLOv11 (Ultralytics) |
| Tracking | ByteTrack |
| Head-pose estimation | MediaPipe Face Mesh |
| Database | SQLite |
| Desktop notifications | Plyer |
| GUI | Tkinter |
| GPU acceleration | CUDA / PyTorch |

---

## Project Structure

```
cheating_detection_system/
├── main_app.py         # Tkinter GUI — the app you run
├── pipeline.py          # Wires detection → cheating rules → screenshot → DB → notify
├── detector.py           # YOLO detection + ByteTrack tracking, box drawing
├── cheating_logic.py     # Cheating rules: phone proximity + sustained head-turn
├── database.py            # SQLite storage for alerts
├── notifier.py             # Desktop notification wrapper
├── config.py                # All tunable constants in one place
├── requirements.txt
├── alerts/
│   ├── alerts.db          # auto-created on first run
│   └── screenshots/       # one image per logged alert
└── models/                # YOLO weights (auto-downloaded on first run)
```

---

## Installation

```bash
git clone https://github.com/Abdo0777/cheating-tracking-system.git
cd cheating-tracking-system

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### GPU setup (recommended)

Detection runs much faster on a CUDA GPU. Check if PyTorch can see yours:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, install the CUDA build matching your driver (check
your version with `nvidia-smi`):

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

```bash
python main_app.py
```

1. Click **Upload Video** and select an exam recording
2. Watch the live annotated preview — green boxes for normal students, red for flagged
3. Every flagged event appears instantly in the **Alert History** table with its behavior and confidence score
4. Select any row and click **View Screenshot** to see exactly what was captured

You can also run the detection pipeline standalone (opens a plain OpenCV
preview window — useful while tuning thresholds):

```bash
python detector.py path/to/video.mp4
```

---

## Configuration

All thresholds live in `config.py`:

| Setting | What it controls |
|---|---|
| `CONF_THRESHOLD` | Minimum detection confidence |
| `PHONE_PROXIMITY_PX` | How close a phone must be to a student to count as "holding it" |
| `HEAD_TURN_FRAMES_THRESHOLD` | Consecutive turned-head frames before flagging |
| `COOLDOWN_SECONDS` | Minimum gap between repeat alerts for the same student |
| `DEVICE` | `0` for GPU, `"cpu"` for CPU-only |

Head-turn sensitivity and occlusion tolerance are in `cheating_logic.py`
(`YAW_RATIO_THRESHOLD`, `PERSON_STALE_SECONDS`).

---

## How It Works

```
Video frame
   │
   ▼
YOLO detection (person, phone) ──▶ ByteTrack (assigns persistent IDs)
   │
   ▼
Cheating rules (cheating_logic.py)
   ├── phone near a tracked student?
   └── head turned away for N frames?
   │
   ▼
Flagged? ──▶ screenshot saved + logged to SQLite + desktop notification
   │
   ▼
Annotated frame (red/green boxes) ──▶ shown live in the GUI
```

---

## Known Limitations & Roadmap

- Phone detection uses YOLO's general COCO "cell phone" class — reliable
  for a clearly visible phone, weaker when it's partly hidden by a hand.
  Fine-tuning on real exam-room footage with a custom "phone" class would
  meaningfully improve this.
- Head-turn detection is a lightweight geometric heuristic (nose position
  relative to face width), not full 3D head-pose estimation.
- Currently local-only (SQLite + local screenshots) by design for easy
  setup. `database.py` isolates all storage calls, so swapping in Firebase
  Firestore/Storage later doesn't require touching the rest of the app.
- No custom app icon yet.

---

## License

MIT — feel free to use, modify, and build on this.
