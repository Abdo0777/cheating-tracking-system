import os
import time
import cv2

from config import SCREENSHOT_DIR
from detector import load_model, track_frame, draw_detections
from cheating_logic import CheatingDetector
from database import init_db, insert_alert
from notifier import notify_cheating


def process_video(video_path, frame_callback=None, stop_flag=None):
    init_db()
    model = load_model()
    cheating_detector = CheatingDetector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    alert_count = 0
    try:
        while True:
            if stop_flag is not None and stop_flag():
                break

            ok, frame = cap.read()
            if not ok:
                break

            detections = track_frame(model, frame)
            cheating_ids, new_events = cheating_detector.update(frame, detections)

            for tid, behavior, conf in new_events:
                ts = int(time.time())
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"track{tid}_{ts}.jpg")
                cv2.imwrite(screenshot_path, frame)
                insert_alert(tid, behavior, conf, screenshot_path)
                notify_cheating(tid, behavior)
                alert_count += 1

            frame = draw_detections(frame, detections, cheating_ids)

            if frame_callback is not None:
                frame_callback(frame)
    finally:
        cap.release()

    return alert_count
