import threading
from datetime import datetime, timedelta
from pathlib import Path

import config
import database

_lock = threading.Lock()


def _init_app():
    import firebase_admin
    from firebase_admin import credentials

    with _lock:
        if not firebase_admin._apps:
            credential = credentials.Certificate(str(config.FIREBASE_CREDENTIALS))
            firebase_admin.initialize_app(credential, {"storageBucket": config.FIREBASE_STORAGE_BUCKET})


def _upload_and_record(alert):
    from firebase_admin import firestore, storage

    _init_app()
    bucket = storage.bucket()
    blob = bucket.blob(f"screenshots/{Path(alert['screenshot_path']).name}")
    blob.upload_from_filename(alert["screenshot_path"])
    try:
        blob.make_public()
        url = blob.public_url
    except Exception:
        url = blob.generate_signed_url(version="v4", expiration=timedelta(days=7))

    firestore.client().collection("alerts").add({
        "created_at": datetime.fromisoformat(alert["created_at"]),
        "student_id": alert["student_id"],
        "behavior": alert["behavior"],
        "confidence": alert["confidence"],
        "screenshot_url": url,
        "video_name": alert["video_name"],
        "video_time": alert["video_time"],
    })
    database.update_firebase_url(alert["id"], url)
    return url


def sync_alert(alert, on_done=None):
    if not config.FIREBASE_ENABLED:
        return

    def work():
        try:
            url = _upload_and_record(alert)
            if on_done:
                on_done(alert["id"], url, None)
        except Exception as error:
            print("Firebase sync failed:", error)
            if on_done:
                on_done(alert["id"], None, str(error))

    threading.Thread(target=work, daemon=True).start()
