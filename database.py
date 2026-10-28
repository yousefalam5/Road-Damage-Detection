import csv
import sqlite3
from contextlib import closing

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    student_id INTEGER,
    behavior TEXT NOT NULL,
    confidence REAL,
    screenshot_path TEXT,
    firebase_url TEXT,
    video_name TEXT,
    video_time TEXT
)
"""


def _connect():
    config.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with closing(_connect()) as connection, connection:
        connection.execute(SCHEMA)


def insert_alert(created_at, student_id, behavior, confidence, screenshot_path, video_name, video_time):
    with closing(_connect()) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO alerts (created_at, student_id, behavior, confidence, screenshot_path, video_name, video_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created_at, student_id, behavior, confidence, str(screenshot_path), video_name, video_time),
        )
        return cursor.lastrowid


def update_firebase_url(alert_id, url):
    with closing(_connect()) as connection, connection:
        connection.execute("UPDATE alerts SET firebase_url = ? WHERE id = ?", (url, alert_id))


def fetch_alerts(limit=None):
    query = "SELECT * FROM alerts ORDER BY id DESC"
    params = ()
    if limit:
        query += " LIMIT ?"
        params = (limit,)
    with closing(_connect()) as connection:
        return [dict(row) for row in connection.execute(query, params).fetchall()]


def clear_alerts():
    with closing(_connect()) as connection, connection:
        connection.execute("DELETE FROM alerts")


def export_csv(path):
    rows = fetch_alerts()
    columns = ["id", "created_at", "student_id", "behavior", "confidence",
               "screenshot_path", "firebase_url", "video_name", "video_time"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
