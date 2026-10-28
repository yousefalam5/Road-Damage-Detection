from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config

NOSE, L_EYE, R_EYE, L_SHOULDER, R_SHOULDER = 0, 1, 2, 5, 6
PHONE = "Phone detected"
HEAD_TURN = "Looking away / head turned"
GAZE = "Suspicious gaze direction"


def head_turn_ratio(points, scores):
    needed = (NOSE, L_SHOULDER, R_SHOULDER)
    if min(scores[i] for i in needed) < config.KEYPOINT_MIN_SCORE:
        return None, 0.0
    width = abs(points[L_SHOULDER][0] - points[R_SHOULDER][0])
    if width < config.MIN_SHOULDER_WIDTH_PX:
        return None, 0.0
    center = (points[L_SHOULDER][0] + points[R_SHOULDER][0]) / 2.0
    ratio = (points[NOSE][0] - center) / width
    score = float(np.mean([scores[i] for i in needed]))
    return float(ratio), score


def head_yaw(points, scores):
    needed = (NOSE, L_EYE, R_EYE)
    if min(scores[i] for i in needed) < config.FACE_MIN_SCORE:
        return None, 0.0
    eye_distance = abs(points[L_EYE][0] - points[R_EYE][0])
    if eye_distance < config.MIN_EYE_DIST_PX:
        return None, 0.0
    eye_center = (points[L_EYE][0] + points[R_EYE][0]) / 2.0
    yaw = (points[NOSE][0] - eye_center) / eye_distance
    return float(yaw), float(np.mean([scores[i] for i in needed]))


def turn_strength(points, scores):
    yaw, yaw_score = head_yaw(points, scores)
    ratio, ratio_score = head_turn_ratio(points, scores)
    candidates = []
    if yaw is not None:
        candidates.append((abs(yaw) / config.YAW_THRESHOLD, yaw_score))
    if ratio is not None:
        candidates.append((abs(ratio) / config.HEAD_TURN_RATIO, ratio_score))
    if not candidates:
        return None, 0.0, yaw, ratio
    strength, score = max(candidates)
    return float(strength), float(score), yaw, ratio


class Baseline:
    def __init__(self):
        self.samples = deque()

    def deviation(self, t, value):
        self.samples.append((t, value))
        while self.samples and t - self.samples[0][0] > config.BASELINE_SECONDS:
            self.samples.popleft()
        if len(self.samples) < config.BASELINE_MIN_SAMPLES:
            return None
        return float(value - np.median([v for _, v in self.samples]))


def has_neighbor_on_side(box, other_boxes, direction):
    if direction == 0:
        return False
    center_x = (box[0] + box[2]) / 2.0
    width = box[2] - box[0]
    for other in other_boxes:
        dx = (other[0] + other[2]) / 2.0 - center_x
        if dx * direction > 0 and abs(dx) <= config.NEIGHBOR_MAX_WIDTHS * width:
            return True
    return False


def is_foreground(box, tallest_height):
    return (box[3] - box[1]) >= config.MIN_RELATIVE_HEIGHT * tallest_height


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def contains_center(person_box, other_box, margin=None):
    margin = config.PHONE_BOX_MARGIN if margin is None else margin
    width = person_box[2] - person_box[0]
    height = person_box[3] - person_box[1]
    cx, cy = _center(other_box)
    return (
        person_box[0] - margin * width <= cx <= person_box[2] + margin * width
        and person_box[1] - margin * height <= cy <= person_box[3] + margin * height
    )


def assign_to_person(person_boxes, object_box):
    best_id, best_distance = None, float("inf")
    ox, oy = _center(object_box)
    for student_id, box in person_boxes.items():
        if not contains_center(box, object_box):
            continue
        px, py = _center(box)
        distance = (ox - px) ** 2 + (oy - py) ** 2
        if distance < best_distance:
            best_id, best_distance = student_id, distance
    return best_id


@dataclass
class StudentState:
    phone_hits: int = 0
    turn_samples: list = field(default_factory=list)
    last_turn_t: Optional[float] = None
    last_alert: dict = field(default_factory=dict)
    red_until: float = -1.0
    last_behavior: str = ""
    yaw_base: Baseline = field(default_factory=Baseline)
    ratio_base: Baseline = field(default_factory=Baseline)
    gaze_base: Baseline = field(default_factory=Baseline)

    def is_suspicious(self, t):
        return t < self.red_until

    def _ready(self, behavior, t):
        return t - self.last_alert.get(behavior, -1e9) >= config.ALERT_COOLDOWN_SECONDS

    def _mark(self, behavior, t):
        self.red_until = max(self.red_until, t + config.RED_HOLD_SECONDS)
        self.last_behavior = behavior

    def _fire(self, behavior, confidence, t):
        self.last_alert[behavior] = t
        self._mark(behavior, t)
        return {"behavior": behavior, "confidence": float(confidence)}

    def update(self, t, phone_conf=None, turn_strength=None, turn_score=0.0, extra=None, turn_label=HEAD_TURN):
        events = []

        if phone_conf is not None:
            self.phone_hits += 1
            if self.phone_hits >= config.PHONE_CONFIRM_FRAMES:
                self._mark(PHONE, t)
                if self._ready(PHONE, t):
                    events.append(self._fire(PHONE, phone_conf, t))
        else:
            self.phone_hits = 0

        if turn_strength is not None:
            dt = 0.0 if self.last_turn_t is None else min(t - self.last_turn_t, 1.0)
            self.last_turn_t = t
            turned = turn_strength >= 1.0
            cutoff = t - config.TURN_WINDOW_SECONDS
            self.turn_samples = [s for s in self.turn_samples if s[0] >= cutoff]
            self.turn_samples.append((t, turned, dt))
            turned_time = sum(sample_dt for _, flag, sample_dt in self.turn_samples if flag)
            if turned and turned_time >= config.HEAD_TURN_SECONDS:
                self._mark(turn_label, t)
                if self._ready(turn_label, t):
                    events.append(self._fire(turn_label, turn_score, t))

        for name, confidence in extra or []:
            self._mark(name, t)
            if self._ready(name, t):
                events.append(self._fire(name, confidence, t))

        return events
