import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")   # swap for a custom-trained weight later
DB_PATH = os.path.join(BASE_DIR, "alerts", "alerts.db")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "alerts", "screenshots")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --- Detection ---
DEVICE = 0          # GPU index for the 3060 — confirmed working
CONF_THRESHOLD = 0.4
TRACKER_CFG = "bytetrack.yaml"   # ships with ultralytics

# Classes we care about from a general model (COCO ids); we'll refine once
# you either fine-tune a model or add custom classes like "phone", "notes"
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67   # "cell phone" in COCO

# --- Cheating rules (tune these as we test) ---
PHONE_PROXIMITY_PX = 80         # phone box center within this distance of a person box -> flag
HEAD_TURN_FRAMES_THRESHOLD = 15  # consecutive frames of suspicious pose before flagging
COOLDOWN_SECONDS = 5             # don't re-log the same student's alert more than once per N sec

# --- Colors (BGR for OpenCV) ---
COLOR_NORMAL = (0, 200, 0)
COLOR_CHEATING = (0, 0, 255)
