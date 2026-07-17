"""
Step 5: Tkinter desktop app tying everything together.
- Upload Video: pick a file, runs the full detection pipeline in a
  background thread so the GUI doesn't freeze.
- Live preview: shows annotated frames (green/red boxes) as it processes.
- Alert History: every logged cheating event, newest first.
- View Screenshot: opens the saved screenshot for the selected alert.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from pipeline import process_video
from database import init_db, get_all_alerts


class CheatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exam Cheating Detection System")
        self.root.geometry("1050x650")

        init_db()

        self.frame_queue = queue.Queue(maxsize=2)
        self.processing = False
        self.stop_requested = False
        self._screenshot_lookup = {}

        self._build_ui()
        self._poll_frame_queue()
        self.refresh_history()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Upload Video", command=self.upload_video).pack(side="left")
        self.status_label = ttk.Label(top, text="Idle")
        self.status_label.pack(side="left", padx=10)

        ttk.Button(top, text="Refresh History", command=self.refresh_history).pack(side="right")
        ttk.Button(top, text="View Screenshot", command=self.view_selected_screenshot).pack(side="right", padx=5)

        body = ttk.Frame(self.root, padding=10)
        body.pack(fill="both", expand=True)

        self.video_label = ttk.Label(body, background="black")
        self.video_label.pack(side="left", fill="both", expand=True)

        history_frame = ttk.Frame(body)
        history_frame.pack(side="right", fill="y")

        columns = ("id", "student", "time", "behavior", "confidence")
        self.tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=27)
        widths = {"id": 40, "student": 60, "time": 140, "behavior": 150, "confidence": 80}
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=widths[col])
        self.tree.pack(fill="y", expand=True)

    # ---------- Actions ----------
    def upload_video(self):
        if self.processing:
            messagebox.showinfo("Busy", "Already processing a video.")
            return

        path = filedialog.askopenfilename(
            title="Select exam video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return

        self.processing = True
        self.stop_requested = False
        self.status_label.config(text=f"Processing: {os.path.basename(path)}")

        threading.Thread(target=self._run_pipeline, args=(path,), daemon=True).start()

    def _run_pipeline(self, path):
        def on_frame(frame):
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # drop frame rather than block processing

        try:
            alert_count = process_video(path, frame_callback=on_frame,
                                         stop_flag=lambda: self.stop_requested)
            self.root.after(0, lambda: self.status_label.config(
                text=f"Done. {alert_count} alert(s) logged."))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {e}"))
        finally:
            self.processing = False
            self.root.after(0, self.refresh_history)

    def refresh_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        self._screenshot_lookup = {}
        for alert_id, track_id, ts, behavior, conf, screenshot in get_all_alerts():
            self.tree.insert("", "end", iid=str(alert_id),
                              values=(alert_id, track_id, ts, behavior, f"{conf:.2f}"))
            self._screenshot_lookup[str(alert_id)] = screenshot

    def view_selected_screenshot(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("No selection", "Select an alert row first.")
            return

        path = self._screenshot_lookup.get(selected[0])
        if not path or not os.path.isfile(path):
            messagebox.showerror("Missing file", "Screenshot file not found.")
            return

        top = tk.Toplevel(self.root)
        top.title(os.path.basename(path))
        img = Image.open(path)
        img.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(img)
        label = ttk.Label(top, image=photo)
        label.image = photo  # keep a reference so it isn't garbage-collected
        label.pack()

    # ---------- Thread-safe frame display ----------
    def _poll_frame_queue(self):
        try:
            frame = self.frame_queue.get_nowait()
            self._display_frame(frame)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_frame_queue)

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((760, 560))
        photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def on_close(self):
        self.stop_requested = True
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CheatingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
