import urllib.request

import cv2
import numpy as np

import config

IMAGE_LEFT_CORNER_R, IMAGE_RIGHT_CORNER_R, IRIS_R = 33, 133, 468
IMAGE_LEFT_CORNER_L, IMAGE_RIGHT_CORNER_L, IRIS_L = 362, 263, 473


def _ratio(left_corner_x, right_corner_x, iris_x):
    width = right_corner_x - left_corner_x
    if abs(width) < 1e-4:
        return None
    return (iris_x - left_corner_x) / width


def gaze_offset_from_landmarks(landmarks):
    if len(landmarks) < 478:
        return None
    ratios = [
        _ratio(landmarks[IMAGE_LEFT_CORNER_R].x, landmarks[IMAGE_RIGHT_CORNER_R].x, landmarks[IRIS_R].x),
        _ratio(landmarks[IMAGE_LEFT_CORNER_L].x, landmarks[IMAGE_RIGHT_CORNER_L].x, landmarks[IRIS_L].x),
    ]
    if any(r is None for r in ratios):
        return None
    return float(np.mean(ratios) - 0.5)


class GazeEstimator:
    def __init__(self):
        import mediapipe as mp

        self.mp = mp
        model_bytes = self._load_model_bytes()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _load_model_bytes():
        path = config.GAZE_MODEL_PATH
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading face landmark model...")
            urllib.request.urlretrieve(config.GAZE_MODEL_URL, str(path))
        return path.read_bytes()

    def estimate(self, frame, box):
        x1, y1, x2, y2 = (int(v) for v in box)
        height = y2 - y1
        top, bottom = max(0, y1), min(frame.shape[0], int(y1 + 0.5 * height))
        left, right = max(0, x1), min(frame.shape[1], x2)
        if bottom - top < 20 or right - left < 20:
            return None
        crop = frame[top:bottom, left:right]
        if crop.shape[0] < 256:
            scale = 256.0 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        if not result.face_landmarks:
            return None
        return gaze_offset_from_landmarks(result.face_landmarks[0])
