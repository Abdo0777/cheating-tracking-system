# Exam Cheating Tracking & Alert System

Desktop app: upload an exam video, it tracks students, flags suspicious
behavior (phone use, sustained head-turn), screenshots the moment, logs it
to a local database, and shows a desktop notification + alert history.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Make sure you have CUDA-enabled torch installed to use your GPU (see
`config.py` -> `DEVICE`). Check with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, reinstall torch from https://pytorch.org/get-started/locally/
with the CUDA build matching your driver (`nvidia-smi` shows the version).

## Run

```bash
python main_app.py
```

Click **Upload Video**, pick an exam recording. It processes in the
background, shows a live annotated preview (green = normal, red =
flagged), and logs alerts into the **Alert History** table on the right.
Select a row and click **View Screenshot** to see the captured frame.

You can also run the detection pipeline standalone (useful for tuning,
opens an OpenCV preview window instead of the full GUI):
```bash
python detector.py your_video.mp4
```

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All tunable constants — thresholds, paths, device, colors |
| `database.py` | SQLite storage for alerts (`alerts.db`) |
| `detector.py` | YOLO detection + ByteTrack tracking, box drawing, CLI test harness |
| `cheating_logic.py` | Cheating rules: phone proximity + sustained head-turn (MediaPipe) |
| `notifier.py` | Desktop notification wrapper (plyer) |
| `pipeline.py` | Wires detection → cheating logic → screenshot → DB → notify, reused by both the CLI and the GUI |
| `main_app.py` | Tkinter GUI — the actual app you run |
| `alerts/screenshots/` | Saved screenshots, one per logged alert |
| `alerts/alerts.db` | SQLite database (auto-created on first run) |

## Tuning (in `config.py`)

- `CONF_THRESHOLD` — detection confidence cutoff (0.4 default)
- `PHONE_PROXIMITY_PX` — how close (pixels) a phone must be to a student to
  count as "holding it" — depends on your camera's resolution/distance,
  tune using the debug prints in `detector.py`'s test loop
- `HEAD_TURN_FRAMES_THRESHOLD` — how many consecutive frames of "turned"
  head before it counts as suspicious (avoids flagging a quick glance)
- `COOLDOWN_SECONDS` — minimum gap between repeat alerts for the same
  student, so one sustained cheat isn't logged 30x/second
- `cheating_logic.py` → `YAW_RATIO_THRESHOLD` and `PERSON_STALE_SECONDS` —
  head-turn sensitivity, and how long to keep using a student's last known
  position if the person detector briefly drops them (e.g. hand covering
  face)

## Known limitations / next steps

- Phone detection uses YOLO's general COCO "cell phone" class, which is
  weak for small/occluded phones. For production accuracy, fine-tune a
  YOLO model on exam-room footage with your own "phone", "notes", etc.
  classes.
- Head-turn detection is a simple geometric heuristic (nose position vs.
  face width), not true 3D head-pose estimation — good enough to flag
  sustained turning, not precise angles.
- Currently local-only (SQLite + local screenshots). To move to Firebase
  Firestore/Storage as in the original spec, only `database.py` needs to
  change — `pipeline.py` and `main_app.py` call it through the same
  `init_db()` / `insert_alert()` / `get_all_alerts()` functions.
- No custom app icon yet — add an `.ico` file and set it via
  `root.iconbitmap("path/to/icon.ico")` in `main_app.py` if wanted.
