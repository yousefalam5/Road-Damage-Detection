# Exam Guard: Cheating Detection Desktop App

Desktop application (Tkinter) that tracks students in an exam video, flags suspicious behavior, turns the suspect's bounding box **red** (normal students stay **green**), saves a screenshot, stores the incident in a database, and shows an immediate desktop alert.

## Demo

<!-- Paste the GitHub video link here (drag the .mp4 into the README editor on GitHub) -->

![App screenshot](docs/screenshot.png)

## Features

- **Detection and tracking:** YOLOv8 pose model with the BoT-SORT tracker (ByteTrack available via `TRACKER = "bytetrack.yaml"`). Each student keeps an ID.
- **Behaviors detected:**
  - *Phone detected:* a phone found on a student for several consecutive processed frames (YOLOv8 object detection).
  - *Looking away / head turned:* the student's head turns toward a neighbor, measured from nose, eye and shoulder keypoints.
  - *Suspicious gaze direction:* iris position (MediaPipe Face Landmarker) moves sideways toward a neighbor even if the head stays still.
  - *Custom behaviors:* optional custom YOLO weights (see below).
- **Boxes:** red for the student being flagged, green for everyone else.
- **Screenshot:** the annotated frame is saved to `alerts/screenshots/`.
- **Database:** every alert is stored in SQLite (`alerts/alerts.db`) with date/time, student ID, behavior, confidence, screenshot path, video name and video time. Optional Firebase Firestore (records) and Storage (screenshots) sync.
- **Desktop app:** Upload Video button, live video with boxes, live alert list, pop-up notification with sound, and an Alert History tab (open screenshot, export CSV, delete records).

Example notification:

```
Cheating Alert Detected
Behavior: Phone detected
Student ID: 3
Time: 10:35 AM
Screenshot saved successfully
```

## How the head and gaze rules work

- Each student is compared with **their own normal position** over the last 30 seconds, so a student who naturally sits slightly turned is not flagged. The first second of each new track is a warm-up.
- A head turn or gaze counts only when it points **toward another monitored student** that is close by.
- An alert needs a total of about 1 second of looking away inside a 6 second window, so repeated short glances are caught.
- Small background students (shorter than 45% of the tallest person) are not monitored.

## Install

Python 3.10 to 3.12. Tkinter ships with Python (on Ubuntu: `sudo apt install python3-tk`).

```bash
pip install -r requirements.txt
python app.py
```

On the first run the app downloads the YOLO weights and the face landmark model automatically (Internet needed once). A GPU is not required, but it is much faster. Lower the load with `PROCESS_EVERY_N_FRAMES` or `IMG_SIZE` in `config.py`.

## Project structure

```
app.py             Tkinter desktop application (entry point)
detector.py        YOLO pose + tracking + object detection, rules, box drawing
behavior.py        Cheating rules and per-student state
gaze.py            MediaPipe iris gaze estimation
pipeline.py        Video loop, screenshot saving, database insert, alert dispatch
database.py        SQLite storage and CSV export
firebase_sync.py   Optional Firebase Firestore + Storage upload
analyze_log.py     Summary of alerts/debug_log.csv for tuning
config.py          All thresholds and paths
```

## Tuning (config.py)

| Setting | Meaning |
|---|---|
| `YAW_THRESHOLD` | Head-turn sensitivity from nose and eye keypoints (0.12). Lower = more sensitive. |
| `HEAD_TURN_RATIO` | Same idea using the shoulders (0.30). |
| `GAZE_THRESHOLD` | Sideways iris offset needed to count as looking away (0.05). `GAZE_ENABLED = False` turns gaze off. |
| `HEAD_TURN_SECONDS`, `TURN_WINDOW_SECONDS` | Total time looking away inside a sliding window (1 s in 6 s). |
| `BASELINE_SECONDS`, `BASELINE_MIN_SAMPLES` | Window used for each student's normal position and warm-up length. |
| `MIN_RELATIVE_HEIGHT` | Students shorter than this fraction of the tallest box are not monitored (0.45). |
| `REQUIRE_NEIGHBOR_DIRECTION` | Count head/gaze only when it points toward another student. |
| `PHONE_CONF`, `PHONE_CONFIRM_FRAMES` | Phone confidence and the number of processed frames that confirm it. |
| `ALERT_COOLDOWN_SECONDS` | Minimum time between two alerts of the same type for the same student. |
| `SHOW_DEBUG` | Shows `value/deviation` for yaw and gaze under each box and writes `alerts/debug_log.csv`. Run `python analyze_log.py` to summarize it. |

The confidence stored for a phone alert is the YOLO detection confidence. For head-turn alerts it is the mean pose keypoint confidence, and for gaze alerts it is a rule score (how far the offset passes the threshold, capped at 1.0).

## Use your own trained YOLO model

If you train a YOLO model on a cheating dataset, put the weights in the project and set:

```python
CUSTOM_MODEL = "weights/best.pt"
CUSTOM_CHEAT_CLASSES = ["cheating", "looking_away"]   # None = every class counts
```

Detections of those classes inside a tracked student's box raise an alert named after the class.

## Firebase (optional)

1. Create a Firebase project and enable **Firestore** and **Storage**.
2. Project settings > Service accounts > Generate new private key. Save it as `firebase_key.json` next to `app.py`.
3. In `config.py`:

```python
FIREBASE_ENABLED = True
FIREBASE_STORAGE_BUCKET = "your-project-id.appspot.com"
```

Alerts are always saved locally first. The upload runs in the background and the cloud URL is written back to the local record. A Firebase failure never stops detection. Never commit `firebase_key.json`.

## Limitations

- The rules are heuristics, not a trained cheating classifier. Results depend on camera angle, face size and video quality; tune the thresholds on your own videos.
- Gaze is estimated from the iris position, which is weak when faces are small or blurry, and eye movement alone can be missed.
- Short clips and scene cuts reset track IDs, which restarts each student's warm-up.
- Phones are only detected when visible to the camera.
- Alerts are suggestions for a human proctor, not proof of cheating. Get consent and follow your institution's privacy rules before recording students.
