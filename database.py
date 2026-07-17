"""
SQLite storage for cheating alerts.
Swap-in note: if you later want Firebase Firestore instead, keep the same
function signatures (init_db, insert_alert, get_all_alerts) so main_app.py
doesn't need to change.
"""
import sqlite3
from datetime import datetime
from config import DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            timestamp TEXT NOT NULL,
            behavior TEXT NOT NULL,
            confidence REAL,
            screenshot_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_alert(track_id: int, behavior: str, confidence: float, screenshot_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alerts (track_id, timestamp, behavior, confidence, screenshot_path) VALUES (?, ?, ?, ?, ?)",
        (track_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), behavior, confidence, screenshot_path)
    )
    conn.commit()
    conn.close()


def get_all_alerts():
    """Returns rows newest-first: (id, track_id, timestamp, behavior, confidence, screenshot_path)"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
