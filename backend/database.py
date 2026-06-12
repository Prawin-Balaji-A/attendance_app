"""
database.py — SQLite + pickle-based storage for attendance backend.

Tables:
  users      : user_id (PK), name, group_name
  attendance : id, user_id, name, group_name, date, time, timestamp

Encodings stored as pickle: Dict[user_id, List[np.ndarray]]
"""

import os
import sqlite3
import pickle
import threading
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'attendance.db')
ENC_PATH = os.path.join(os.path.dirname(__file__), 'data', 'encodings.pkl')

_lock = threading.Lock()

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock:
        c = _conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id   TEXT PRIMARY KEY,
                name      TEXT NOT NULL,
                group_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                name       TEXT NOT NULL,
                group_name TEXT NOT NULL,
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                timestamp  TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS
                attendance_user_date ON attendance(user_id, date);
        """)
        c.commit()
        c.close()


# ── Users ──────────────────────────────────────────────────────────────────

def add_user(user_id: str, name: str, group: str):
    with _lock:
        c = _conn()
        c.execute(
            "INSERT OR REPLACE INTO users(user_id, name, group_name) VALUES(?,?,?)",
            (user_id, name, group),
        )
        c.commit()
        c.close()


def get_users():
    with _lock:
        c = _conn()
        rows = c.execute("SELECT user_id, name, group_name FROM users ORDER BY name").fetchall()
        c.close()
    return [{"user_id": r["user_id"], "name": r["name"], "group": r["group_name"]} for r in rows]


def delete_user(user_id: str) -> bool:
    with _lock:
        c = _conn()
        cur = c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        c.commit()
        affected = cur.rowcount
        c.close()

    # Remove encodings
    encodings = get_encodings()
    if user_id in encodings:
        del encodings[user_id]
        save_encodings(encodings)

    return affected > 0


# ── Encodings ──────────────────────────────────────────────────────────────

def get_encodings() -> dict:
    with _lock:
        if not os.path.exists(ENC_PATH):
            return {}
        with open(ENC_PATH, 'rb') as f:
            return pickle.load(f)


def save_encodings(encodings: dict):
    with _lock:
        with open(ENC_PATH, 'wb') as f:
            pickle.dump(encodings, f)


# ── Attendance ─────────────────────────────────────────────────────────────

def already_logged_today(user_id: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        c = _conn()
        row = c.execute(
            "SELECT 1 FROM attendance WHERE user_id=? AND date=?",
            (user_id, today),
        ).fetchone()
        c.close()
    return row is not None


def log_attendance(user_id: str, name: str, group: str) -> bool:
    """Returns True if newly logged, False if already logged today."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

    with _lock:
        c = _conn()
        try:
            c.execute(
                """INSERT OR IGNORE INTO attendance
                   (user_id, name, group_name, date, time, timestamp)
                   VALUES(?,?,?,?,?,?)""",
                (user_id, name, group, date_str, time_str, ts_str),
            )
            c.commit()
            affected = c.total_changes
        except Exception:
            affected = 0
        finally:
            c.close()
    return affected > 0


def get_attendance():
    with _lock:
        c = _conn()
        rows = c.execute(
            "SELECT user_id, name, group_name, date, time, timestamp "
            "FROM attendance ORDER BY timestamp DESC"
        ).fetchall()
        c.close()
    return [
        {
            "UserID": r["user_id"],
            "Name": r["name"],
            "Group": r["group_name"],
            "Date": r["date"],
            "Time": r["time"],
            "Timestamp": r["timestamp"],
        }
        for r in rows
    ]


def get_groups():
    """Return groups with today's present/absent counts for the dashboard."""
    today = datetime.now().strftime("%Y-%m-%d")

    with _lock:
        c = _conn()
        # All distinct groups from users table
        groups_rows = c.execute(
            "SELECT DISTINCT group_name FROM users"
        ).fetchall()

        results = []
        for gr in groups_rows:
            gname = gr["group_name"]

            total = c.execute(
                "SELECT COUNT(*) as cnt FROM users WHERE group_name=?", (gname,)
            ).fetchone()["cnt"]

            present = c.execute(
                """SELECT COUNT(*) as cnt FROM attendance a
                   JOIN users u ON a.user_id = u.user_id
                   WHERE u.group_name=? AND a.date=?""",
                (gname, today),
            ).fetchone()["cnt"]

            absent = total - present

            results.append({
                "groupName": gname,
                "totalMembers": total,
                "presentCount": present,
                "absentCount": absent,
            })
        c.close()
    return results


def get_group_details(group_name: str):
    today = datetime.now().strftime("%Y-%m-%d")

    with _lock:
        c = _conn()

        all_users = c.execute(
            "SELECT user_id, name, group_name FROM users WHERE group_name=?",
            (group_name,),
        ).fetchall()

        present_ids = set(
            r["user_id"]
            for r in c.execute(
                """SELECT a.user_id FROM attendance a
                   JOIN users u ON a.user_id=u.user_id
                   WHERE u.group_name=? AND a.date=?""",
                (group_name, today),
            ).fetchall()
        )
        c.close()

    present_users = []
    absent_users = []
    for u in all_users:
        entry = {"user_id": u["user_id"], "name": u["name"], "group": u["group_name"]}
        if u["user_id"] in present_ids:
            present_users.append(entry)
        else:
            absent_users.append(entry)

    return {
        "groupName": group_name,
        "totalMembers": len(all_users),
        "presentCount": len(present_users),
        "absentCount": len(absent_users),
        "presentUsers": present_users,
        "absentUsers": absent_users,
    }
