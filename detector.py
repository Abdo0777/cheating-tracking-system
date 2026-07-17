"""
Step 2: person + phone detection with persistent tracking IDs.
Run this file directly to sanity-check tracking on a video before wiring
in the cheating logic (Step 3).
"""
import time
import cv2
from ultralytics import YOLO
from config import (
    MODEL_PATH, DEVICE, CONF_THRESHOLD, TRACKER_CFG,
    PERSON_CLASS_ID, PHONE_CLASS_ID, COLOR_NORMAL, COLOR_CHEATING
)


def load_model():
    return YOLO(MODEL_PATH)


def track_frame(model, frame):
    """
    Runs detection+tracking on one frame.
    Returns a list of dicts: {track_id, class_id, bbox (x1,y1,x2,y2), conf}
    """
    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CFG,
        conf=CONF_THRESHOLD,
        classes=[PERSON_CLASS_ID, PHONE_CLASS_ID],
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(boxes, ids, confs, cls_ids):
            detections.append({
                "track_id": int(tid),
                "class_id": int(cls_id),
                "bbox": tuple(box.astype(int)),
                "conf": float(conf),
            })
    return detections


def draw_detections(frame, detections, cheating_ids=None):
    """Draws boxes: red for track_ids in cheating_ids, green otherwise.
    Phone boxes are drawn yellow just as a visual marker (not part of the
    student color rule)."""
    cheating_ids = cheating_ids or set()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid = det["track_id"]

        if det["class_id"] == PERSON_CLASS_ID:
            color = COLOR_CHEATING if tid in cheating_ids else COLOR_NORMAL
            label = f"Student {tid}" + (" - CHEATING" if tid in cheating_ids else "")
        else:
            color = (0, 220, 220)  # yellow-ish, phone
            label = f"Phone {tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


if __name__ == "__main__":
    import os
    import sys
    import cv2 as _cv2  # already imported above, alias avoids shadow confusion
    from cheating_logic import CheatingDetector
    from config import SCREENSHOT_DIR

    video_path = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = webcam

    if video_path != 0:
        abs_path = os.path.abspath(video_path)
        print(f"Looking for video at: {abs_path}")
        if not os.path.isfile(abs_path):
            print("ERROR: file does not exist at that path. Check the filename/location.")
            sys.exit(1)

    print("Loading model (first run downloads weights, needs internet)...")
    model = load_model()
    cheating_detector = CheatingDetector()
    print("Model loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video source: {video_path}")
        print("This usually means a codec issue or a corrupted/empty file.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened. Reported frame count: {total_frames}")

    frame_num = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"Stream ended (or first read failed) at frame {frame_num}.")
            break

        frame_num += 1
        detections = track_frame(model, frame)

        phone_dets = [d for d in detections if d["class_id"] == PHONE_CLASS_ID]
        if phone_dets:
            for d in phone_dets:
                print(f"[debug] phone seen: conf={d['conf']:.2f} bbox={d['bbox']}")
        elif frame_num % 30 == 0:
            print(f"[debug] frame {frame_num}: no phone detected")

        cheating_ids, new_events = cheating_detector.update(frame, detections)

        for tid, behavior, conf in new_events:
            ts = int(time.time())
            path = os.path.join(SCREENSHOT_DIR, f"track{tid}_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"ALERT: Student {tid} - {behavior} (conf {conf:.2f}) -> {path}")

        frame = draw_detections(frame, detections, cheating_ids)

        cv2.imshow("Tracking test - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Processed {frame_num} frames total.")
    cap.release()
    cv2.destroyAllWindows()
