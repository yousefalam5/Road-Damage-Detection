import csv
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "alerts" / "debug_log.csv"
if not path.exists():
    print("File not found:", path)
    print("Run python app.py, upload the video, let it finish, then run this again.")
    sys.exit(1)

stats = {}
with open(path, newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        s = stats.setdefault(row["student_id"], {"rows": 0, "t_min": 1e9, "t_max": -1.0,
                                                  "gaze": [], "yaw": [], "over": 0, "gaze_missing": 0})
        s["rows"] += 1
        t = float(row["time_s"])
        s["t_min"], s["t_max"] = min(s["t_min"], t), max(s["t_max"], t)
        if row["gaze"]:
            s["gaze"].append(abs(float(row["gaze"])))
        else:
            s["gaze_missing"] += 1
        if row["yaw"]:
            s["yaw"].append(abs(float(row["yaw"])))
        if row["final_strength"] and float(row["final_strength"]) >= 1.0:
            s["over"] += 1

print(f"{'ID':>4} {'rows':>6} {'first_s':>8} {'last_s':>8} {'max_gaze':>9} {'max_yaw':>8} {'gaze_n/a':>9} {'over_thr':>9}")
for student_id, s in sorted(stats.items(), key=lambda kv: -kv[1]["rows"]):
    max_gaze = f"{max(s['gaze']):.3f}" if s["gaze"] else "-"
    max_yaw = f"{max(s['yaw']):.3f}" if s["yaw"] else "-"
    print(f"{student_id:>4} {s['rows']:>6} {s['t_min']:>8.1f} {s['t_max']:>8.1f} {max_gaze:>9} {max_yaw:>8} "
          f"{s['gaze_missing']:>9} {s['over']:>9}")
