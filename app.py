import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config
import database

VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]


class Toast(tk.Toplevel):
    active = []

    def __init__(self, master, lines, duration_ms=6000):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        box = tk.Frame(self, bg="#b00020", padx=16, pady=12)
        box.pack()
        tk.Label(box, text=lines[0], bg="#b00020", fg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        for line in lines[1:]:
            tk.Label(box, text=line, bg="#b00020", fg="white", font=("Segoe UI", 10)).pack(anchor="w")
        self.update_idletasks()
        width, height = self.winfo_width(), self.winfo_height()
        x = self.winfo_screenwidth() - width - 24
        y = self.winfo_screenheight() - height - 90 - sum(t.winfo_height() + 8 for t in Toast.active)
        self.geometry(f"+{x}+{max(y, 10)}")
        Toast.active.append(self)
        self.after(duration_ms, self.close)

    def close(self):
        if self in Toast.active:
            Toast.active.remove(self)
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exam Cheating Detection System")
        self.geometry("1280x780")
        self.minsize(1000, 660)
        database.init_db()

        self.frame_queue = queue.Queue(maxsize=2)
        self.event_queue = queue.Queue()
        self.detector = None
        self.worker = None
        self.stop_event = threading.Event()
        self.photo = None
        self.session_alerts = 0
        self.history_rows = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_history()
        self.after(30, self._poll)

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        live = ttk.Frame(notebook)
        history = ttk.Frame(notebook)
        notebook.add(live, text="  Live Monitoring  ")
        notebook.add(history, text="  Alert History  ")

        toolbar = ttk.Frame(live)
        toolbar.pack(fill="x", padx=10, pady=8)
        self.btn_upload = ttk.Button(toolbar, text="Upload Video", command=self.upload_video)
        self.btn_upload.pack(side="left")
        self.btn_stop = ttk.Button(toolbar, text="Stop", command=self.stop_processing, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.file_label = ttk.Label(toolbar, text="No video selected")
        self.file_label.pack(side="left", padx=10)

        body = ttk.Frame(live)
        body.pack(fill="both", expand=True, padx=10)
        self.canvas = tk.Canvas(body, bg="black", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw_placeholder())

        side = ttk.Frame(body, width=360)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        self.alert_title = ttk.Label(side, text="Live Alerts (0)", font=("Segoe UI", 11, "bold"))
        self.alert_title.pack(anchor="w")
        self.live_list = tk.Listbox(side, activestyle="none", font=("Segoe UI", 10))
        self.live_list.pack(fill="both", expand=True, pady=(4, 0))

        footer = ttk.Frame(live)
        footer.pack(fill="x", padx=10, pady=8)
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.pack(fill="x")
        self.status = ttk.Label(footer, text="Ready. Upload an exam video to start.")
        self.status.pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(history)
        buttons.pack(fill="x", padx=10, pady=8)
        ttk.Button(buttons, text="Refresh", command=self.refresh_history).pack(side="left")
        ttk.Button(buttons, text="Open Screenshot", command=self.open_selected).pack(side="left", padx=6)
        ttk.Button(buttons, text="Export CSV", command=self.export_csv).pack(side="left")
        ttk.Button(buttons, text="Delete All Records", command=self.delete_all).pack(side="right")

        columns = ("id", "time", "student", "behavior", "confidence", "video_time", "screenshot", "cloud")
        headings = ("ID", "Date / Time", "Student", "Behavior", "Confidence", "Video Time", "Screenshot", "Cloud URL")
        widths = (50, 150, 70, 190, 90, 90, 330, 200)
        frame = ttk.Frame(history)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for column, heading, width in zip(columns, headings, widths):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda _e: self.open_selected())

    def _draw_placeholder(self):
        if self.photo is None and (self.worker is None or not self.worker.is_alive()):
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                                    text="Upload an exam video to start", fill="white", font=("Segoe UI", 14))

    def upload_video(self):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Busy", "A video is already being processed. Stop it first.")
            return
        path = filedialog.askopenfilename(title="Select exam video", filetypes=VIDEO_TYPES)
        if not path:
            return
        self.file_label.config(text=Path(path).name)
        self.live_list.delete(0, "end")
        self.session_alerts = 0
        self.alert_title.config(text="Live Alerts (0)")
        self.progress["value"] = 0
        self.stop_event = threading.Event()
        self.btn_upload.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.worker = threading.Thread(target=self._worker, args=(path, self.stop_event), daemon=True)
        self.worker.start()

    def stop_processing(self):
        self.stop_event.set()
        self.status.config(text="Stopping...")

    def _worker(self, path, stop_event):
        try:
            if self.detector is None:
                self.event_queue.put(("status", "Loading detection models (first run downloads weights)..."))
                from detector import CheatingDetector
                self.detector = CheatingDetector()
            else:
                self.detector.reset()
            from pipeline import VideoProcessor

            processor = VideoProcessor(
                path, self.detector, on_frame=self._push_frame,
                on_alert=lambda alert: self.event_queue.put(("alert", alert)),
                stop_event=stop_event,
                on_sync=lambda alert_id, url, error: self.event_queue.put(("sync", alert_id, url, error)),
            )
            self.event_queue.put(("status", "Processing video..."))
            processor.run()
            self.event_queue.put(("done", stop_event.is_set()))
        except Exception as error:
            self.event_queue.put(("error", f"{type(error).__name__}: {error}"))

    def _push_frame(self, frame, index, total):
        item = (frame, index, total)
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(item)
            except queue.Full:
                pass

    def _poll(self):
        while True:
            try:
                message = self.event_queue.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "status":
                self.status.config(text=message[1])
            elif kind == "alert":
                self._on_alert(message[1])
            elif kind == "sync":
                if message[3] is None:
                    self.status.config(text="Alert uploaded to Firebase.")
                    self.refresh_history()
                else:
                    self.status.config(text="Firebase upload failed (alert is saved locally).")
            elif kind == "done":
                self.status.config(text="Stopped." if message[1] else "Finished processing the video.")
                self._finish()
            elif kind == "error":
                messagebox.showerror("Error", message[1])
                self.status.config(text="Error. See message.")
                self._finish()

        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._show_frame(*latest)
        self.after(25, self._poll)

    def _finish(self):
        self.btn_upload.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _show_frame(self, frame, index, total):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10:
            return
        frame_h, frame_w = frame.shape[:2]
        scale = min(width / frame_w, height / frame_h)
        resized = cv2.resize(frame, (max(1, int(frame_w * scale)), max(1, int(frame_h * scale))))
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.photo)
        if total:
            self.progress["value"] = 100.0 * index / total

    def _on_alert(self, alert):
        self.session_alerts += 1
        self.alert_title.config(text=f"Live Alerts ({self.session_alerts})")
        created = datetime.fromisoformat(alert["created_at"])
        self.live_list.insert(
            0, f"{created:%H:%M:%S} | ID {alert['student_id']} | {alert['behavior']} ({alert['confidence']:.0%})")
        self.live_list.itemconfig(0, fg="#b00020")
        self._beep()
        Toast(self, [
            "Cheating Alert Detected",
            f"Behavior: {alert['behavior']}",
            f"Student ID: {alert['student_id']}",
            f"Time: {created:%I:%M %p}",
            "Screenshot saved successfully",
        ])
        self.refresh_history()

    def _beep(self):
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except ImportError:
            self.bell()

    def refresh_history(self):
        self.tree.delete(*self.tree.get_children())
        self.history_rows = {}
        for row in database.fetch_alerts():
            confidence = row["confidence"]
            item = self.tree.insert("", "end", values=(
                row["id"], row["created_at"].replace("T", " "), row["student_id"], row["behavior"],
                f"{confidence:.0%}" if confidence is not None else "-", row["video_time"] or "-",
                row["screenshot_path"], row["firebase_url"] or "-",
            ))
            self.history_rows[item] = row

    def open_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Alert History", "Select an alert first.")
            return
        row = self.history_rows[selection[0]]
        path = row["screenshot_path"]
        if not path or not os.path.exists(path):
            messagebox.showwarning("Screenshot", "Screenshot file not found on this computer.")
            return
        window = tk.Toplevel(self)
        window.title(f"Alert #{row['id']} - {row['behavior']}")
        image = Image.open(path)
        image.thumbnail((1000, 640))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(window, image=photo)
        label.image = photo
        label.pack()
        info = f"Student {row['student_id']} | {row['created_at'].replace('T', ' ')} | video time {row['video_time']}"
        ttk.Label(window, text=info).pack(pady=6)

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            count = database.export_csv(path)
            messagebox.showinfo("Export", f"Exported {count} alerts.")

    def delete_all(self):
        if messagebox.askyesno("Delete", "Delete all alert records from the database?\n(Screenshot files stay on disk.)"):
            database.clear_alerts()
            self.refresh_history()

    def _on_close(self):
        self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
