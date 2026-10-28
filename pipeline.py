import re
from datetime import datetime
from pathlib import Path

import cv2

import config
import database
import firebase_sync


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _format_video_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


class VideoProcessor:
    def __init__(self, video_path, detector, on_frame, on_alert, stop_event, on_sync=None):
        self.video_path = str(video_path)
        self.detector = detector
        self.on_frame = on_frame
        self.on_alert = on_alert
        self.on_sync = on_sync
        self.stop_event = stop_event

    def _save_alert(self, event, t):
        config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        file_name = f"alert_{now:%Y%m%d_%H%M%S_%f}_id{event['student_id']}_{_slug(event['behavior'])}.jpg"
        path = config.SCREENSHOT_DIR / file_name
        if not cv2.imwrite(str(path), event["frame"]):
            raise IOError(f"could not write screenshot: {path}")

        alert = {
            "created_at": now.isoformat(timespec="seconds"),
            "student_id": event["student_id"],
            "behavior": event["behavior"],
            "confidence": round(event["confidence"], 4),
            "screenshot_path": str(path),
            "video_name": Path(self.video_path).name,
            "video_time": _format_video_time(t),
            "firebase_url": None,
        }
        alert["id"] = database.insert_alert(
            alert["created_at"], alert["student_id"], alert["behavior"], alert["confidence"],
            alert["screenshot_path"], alert["video_name"], alert["video_time"],
        )
        firebase_sync.sync_alert(alert, self.on_sync)
        return alert

    def run(self):
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise IOError(f"cannot open video: {self.video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        index = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                t = index / fps
                if index % config.PROCESS_EVERY_N_FRAMES == 0:
                    output, events = self.detector.process_frame(frame, t)
                else:
                    output, events = self.detector.render(frame, t), []
                for event in events:
                    self.on_alert(self._save_alert(event, t))
                self.on_frame(output, index, total)
                index += 1
        finally:
            capture.release()
