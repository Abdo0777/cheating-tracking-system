import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt") 
DB_PATH = os.path.join(BASE_DIR, "alerts", "alerts.db")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "alerts", "screenshots")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

DEVICE = 0  
CONF_THRESHOLD = 0.4
TRACKER_CFG = "bytetrack.yaml"   

PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67   # "cell phone" in COCO

PHONE_PROXIMITY_PX = 80       
HEAD_TURN_FRAMES_THRESHOLD = 15 
COOLDOWN_SECONDS = 5          

COLOR_NORMAL = (0, 200, 0)
COLOR_CHEATING = (0, 0, 255)
