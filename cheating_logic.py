import time
import mediapipe as mp

from config import (
    PERSON_CLASS_ID, PHONE_CLASS_ID, PHONE_PROXIMITY_PX,
    HEAD_TURN_FRAMES_THRESHOLD, COOLDOWN_SECONDS
)

PERSON_STALE_SECONDS = 1.5  # keep using a student's last known box for this long
                             # if the person detector momentarily loses them
                             # (e.g. head down / hand covering face while texting)

_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices used for a simple yaw estimate
_NOSE_TIP = 1
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454

YAW_RATIO_THRESHOLD = 0.35  # deviation from center (0.5) that counts as "turned"


class CheatingDetector:
    def __init__(self):
        self._head_turn_streak = {}   # track_id -> consecutive frame count
        self._last_alert_time = {}    # track_id -> unix time of last logged alert
        self._last_person_seen = {}   # track_id -> (bbox, unix time last seen)

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _estimate_yaw_ratio(frame, bbox):
        """Returns nose horizontal position as a 0-1 ratio across face width
        (0.5 = centered/facing forward). None if no face found in the crop."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        rgb = crop[:, :, ::-1]
        result = _mp_face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        nose_x = lm[_NOSE_TIP].x
        left_x = lm[_LEFT_CHEEK].x
        right_x = lm[_RIGHT_CHEEK].x
        face_width = abs(right_x - left_x)
        if face_width < 1e-4:
            return None
        return (nose_x - min(left_x, right_x)) / face_width

    def _cooldown_ok(self, tid, now):
        last = self._last_alert_time.get(tid)
        return last is None or (now - last) >= COOLDOWN_SECONDS

    def update(self, frame, detections):
        """
        Call once per frame with the frame image and this frame's detections.
        Returns:
            cheating_ids: set of track_ids currently flagged (for box coloring)
            new_events: list of (track_id, behavior, confidence) that just
                        crossed the alert threshold and passed cooldown -
                        these are the ones to screenshot + log to DB.
        """
        persons = [d for d in detections if d["class_id"] == PERSON_CLASS_ID]
        phones = [d for d in detections if d["class_id"] == PHONE_CLASS_ID]

        cheating_ids = set()
        new_events = []
        now = time.time()

        # Update last-known positions for everyone seen this frame
        for person in persons:
            self._last_person_seen[person["track_id"]] = (person["bbox"], now)

        # Build the candidate list for phone matching: current-frame persons
        # plus recently-seen-but-currently-missing ones (bridges occlusion,
        # e.g. a hand covering the face while holding a phone)
        candidates = {p["track_id"]: p["bbox"] for p in persons}
        for tid, (bbox, last_seen) in self._last_person_seen.items():
            if tid not in candidates and (now - last_seen) <= PERSON_STALE_SECONDS:
                candidates[tid] = bbox

        phone_flagged = {}  # track_id -> best confidence
        for phone in phones:
            phone_center = self._center(phone["bbox"])
            for tid, bbox in candidates.items():
                person_center = self._center(bbox)
                if self._distance(phone_center, person_center) <= PHONE_PROXIMITY_PX:
                    phone_flagged[tid] = max(phone_flagged.get(tid, 0), phone["conf"])

        for tid, conf in phone_flagged.items():
            cheating_ids.add(tid)
            if self._cooldown_ok(tid, now):
                new_events.append((tid, "Phone detected", conf))
                self._last_alert_time[tid] = now

        for person in persons:
            tid = person["track_id"]
            ratio = self._estimate_yaw_ratio(frame, person["bbox"])
            turned = ratio is not None and (
                ratio < (0.5 - YAW_RATIO_THRESHOLD) or ratio > (0.5 + YAW_RATIO_THRESHOLD)
            )

            if turned:
                self._head_turn_streak[tid] = self._head_turn_streak.get(tid, 0) + 1
            else:
                self._head_turn_streak[tid] = 0

            if self._head_turn_streak[tid] >= HEAD_TURN_FRAMES_THRESHOLD:
                cheating_ids.add(tid)
                if self._cooldown_ok(tid, now):
                    new_events.append((tid, "Sustained head turn", person["conf"]))
                    self._last_alert_time[tid] = now

        return cheating_ids, new_events
