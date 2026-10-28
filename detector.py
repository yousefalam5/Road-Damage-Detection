import csv

import cv2
import numpy as np

import config
from behavior import GAZE, HEAD_TURN, StudentState, assign_to_person, has_neighbor_on_side, head_turn_ratio, head_yaw, is_foreground

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label_from_class(name):
    return name.replace("_", " ").strip().capitalize() + " detected"


class CheatingDetector:
    def __init__(self):
        from ultralytics import YOLO

        self.pose_model = YOLO(config.POSE_MODEL)
        self.object_model = YOLO(config.OBJECT_MODEL)
        self.custom_model = YOLO(config.CUSTOM_MODEL) if config.CUSTOM_MODEL else None
        self.gaze = None
        if config.GAZE_ENABLED:
            try:
                from gaze import GazeEstimator
                self.gaze = GazeEstimator()
            except Exception as error:
                print("Gaze estimation disabled:", error)
        self.reset()

    def reset(self):
        self.states = {}
        self.tracks = []
        self.object_boxes = []
        self._fresh = True
        old_file = getattr(self, "_log_file", None)
        if old_file is not None:
            try:
                old_file.close()
            except Exception:
                pass
        self._log_file = None
        self._log_writer = None
        if config.SHOW_DEBUG:
            try:
                config.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
                self._log_file = open(config.ALERTS_DIR / "debug_log.csv", "w", newline="", encoding="utf-8")
                self._log_writer = csv.writer(self._log_file)
                self._log_writer.writerow(["time_s", "student_id", "yaw", "gaze", "pose_strength",
                                           "gaze_strength", "final_strength", "yaw_dev", "gaze_dev"])
            except OSError as error:
                print("Debug log disabled:", error)
                self._log_file = None
                self._log_writer = None

    def _log(self, t, student_id, yaw, gaze_offset, pose_strength, gaze_strength, strength, yaw_dev, gaze_dev):
        if self._log_writer is None:
            return
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        try:
            self._log_writer.writerow([f"{t:.2f}", student_id, fmt(yaw), fmt(gaze_offset), fmt(pose_strength),
                                       fmt(gaze_strength), fmt(strength), fmt(yaw_dev), fmt(gaze_dev)])
            self._log_file.flush()
        except OSError:
            self._log_writer = None

    def _track_people(self, frame):
        result = self.pose_model.track(
            frame, persist=not self._fresh, tracker=config.TRACKER, conf=config.PERSON_CONF,
            imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False,
        )[0]
        self._fresh = False
        tracks = []
        if result.boxes is None or len(result.boxes) == 0:
            return tracks
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else list(range(len(boxes)))
        points = scores = None
        if result.keypoints is not None:
            points = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                scores = result.keypoints.conf.cpu().numpy()
        for i, (student_id, box) in enumerate(zip(ids, boxes)):
            tracks.append({
                "id": int(student_id),
                "box": box,
                "points": points[i] if points is not None and scores is not None else None,
                "scores": scores[i] if scores is not None else None,
            })
        return tracks

    def _detect_objects(self, frame):
        found = []
        result = self.object_model.predict(
            frame, conf=config.PHONE_CONF, classes=[config.PHONE_CLASS_ID],
            imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False,
        )[0]
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            found.append(("phone", box, float(conf)))

        if self.custom_model is not None:
            result = self.custom_model.predict(
                frame, conf=config.CUSTOM_CONF, imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False,
            )[0]
            for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy()):
                name = result.names[int(cls)]
                if config.CUSTOM_CHEAT_CLASSES is None or name in config.CUSTOM_CHEAT_CLASSES:
                    found.append((name, box, float(conf)))
        return found

    def process_frame(self, frame, t):
        tracks = self._track_people(frame)
        objects = self._detect_objects(frame)
        tallest = max((tr["box"][3] - tr["box"][1] for tr in tracks), default=0.0)
        monitored = [tr for tr in tracks if is_foreground(tr["box"], tallest)]
        monitored_ids = {tr["id"] for tr in monitored}
        for track in tracks:
            if track["id"] not in monitored_ids:
                track["debug"] = "background (not monitored)"
        person_boxes = {track["id"]: track["box"] for track in monitored}

        phone_conf, extras = {}, {}
        for name, box, conf in objects:
            student_id = assign_to_person(person_boxes, box)
            if student_id is None:
                continue
            if name == "phone":
                phone_conf[student_id] = max(phone_conf.get(student_id, 0.0), conf)
            else:
                extras.setdefault(student_id, []).append((_label_from_class(name), conf))

        events = []
        for track in monitored:
            student_id = track["id"]
            state = self.states.setdefault(student_id, StudentState())
            yaw = ratio = None
            yaw_score = ratio_score = 0.0
            if track["points"] is not None:
                yaw, yaw_score = head_yaw(track["points"], track["scores"])
                ratio, ratio_score = head_turn_ratio(track["points"], track["scores"])
            gaze_offset = None
            gaze_engine = getattr(self, "gaze", None)
            if gaze_engine is not None:
                gaze_offset = gaze_engine.estimate(frame, track["box"])

            yaw_dev = state.yaw_base.deviation(t, yaw) if yaw is not None else None
            ratio_dev = state.ratio_base.deviation(t, ratio) if ratio is not None else None
            gaze_dev = state.gaze_base.deviation(t, gaze_offset) if gaze_offset is not None else None

            candidates = []
            pose_strength = gaze_strength = None
            if yaw_dev is not None:
                value = abs(yaw_dev) / config.YAW_THRESHOLD
                pose_strength = value
                candidates.append((value, yaw_score, HEAD_TURN, 1 if yaw_dev > 0 else -1))
            if ratio_dev is not None:
                value = abs(ratio_dev) / config.HEAD_TURN_RATIO
                pose_strength = value if pose_strength is None else max(pose_strength, value)
                candidates.append((value, ratio_score, HEAD_TURN, 1 if ratio_dev > 0 else -1))
            if gaze_dev is not None:
                gaze_strength = abs(gaze_dev) / config.GAZE_THRESHOLD
                candidates.append((gaze_strength, min(1.0, gaze_strength / 2.0), GAZE, 1 if gaze_dev > 0 else -1))
            if candidates:
                strength, score, label, direction = max(candidates, key=lambda c: c[0])
            else:
                strength, score, label, direction = None, 0.0, HEAD_TURN, 0
            if strength is not None and strength >= 1.0 and config.REQUIRE_NEIGHBOR_DIRECTION:
                others = [other["box"] for other in monitored if other["id"] != student_id]
                if not has_neighbor_on_side(track["box"], others, direction):
                    strength = 0.0
            self._log(t, student_id, yaw, gaze_offset, pose_strength, gaze_strength, strength, yaw_dev, gaze_dev)

            def show(value, deviation):
                if value is None:
                    return "n/a"
                return f"{value:+.2f}/" + (f"{deviation:+.2f}" if deviation is not None else "..")
            track["debug"] = f"yaw {show(yaw, yaw_dev)} | gaze {show(gaze_offset, gaze_dev)}"
            for event in state.update(t, phone_conf.get(student_id), strength, score,
                                      extras.get(student_id), turn_label=label):
                event["student_id"] = student_id
                events.append(event)

        self.tracks = tracks
        self.object_boxes = [box for _, box, _ in objects]
        annotated = self.render(frame, t)
        for event in events:
            event["frame"] = annotated.copy()
        return annotated, events

    def render(self, frame, t):
        output = frame.copy()
        scale = max(0.5, output.shape[1] / 1280 * 0.7)
        thickness = max(2, int(scale * 3))

        for box in self.object_boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(output, (x1, y1), (x2, y2), config.ORANGE, 2)

        for track in self.tracks:
            state = self.states.get(track["id"])
            suspicious = state is not None and state.is_suspicious(t)
            color = config.RED if suspicious else config.GREEN
            x1, y1, x2, y2 = (int(v) for v in track["box"])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            label = f"ID {track['id']}"
            if suspicious and state.last_behavior:
                label += f" | {state.last_behavior}"
            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, scale, 1)
            top = y1 - text_h - baseline - 6
            if top < 0:
                top = y1 + 2
            cv2.rectangle(output, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 6), color, -1)
            cv2.putText(output, label, (x1 + 4, top + text_h + 2), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
            if config.SHOW_DEBUG and track.get("debug"):
                cv2.putText(output, track["debug"], (x1 + 4, min(y2 + text_h + 6, output.shape[0] - 4)),
                            FONT, scale, color, 1, cv2.LINE_AA)
        return output
