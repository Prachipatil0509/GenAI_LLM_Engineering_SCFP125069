import sqlite3
import os
import pandas as pd

# Use local directory for database file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "lab_booking.db")

def get_connection():
    """Establish connection to SQLite database with foreign keys enabled."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(force_reset: bool = False):
    """
    Initialize SQLite tables and seed initial diagnostic tests and slots if not present.
    If force_reset is True, drop existing tables and re-populate fresh data.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if force_reset:
        cursor.execute("DROP TABLE IF EXISTS bookings")
        cursor.execute("DROP TABLE IF EXISTS slots")
        cursor.execute("DROP TABLE IF EXISTS tests")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tests (
        test_id INTEGER PRIMARY KEY,
        test_name TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS slots (
        slot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id INTEGER NOT NULL,
        slot_time TEXT NOT NULL,
        is_booked INTEGER DEFAULT 0,
        FOREIGN KEY(test_id) REFERENCES tests(test_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        test_id INTEGER NOT NULL,
        slot_id INTEGER NOT NULL,
        status TEXT DEFAULT 'confirmed',
        FOREIGN KEY(test_id) REFERENCES tests(test_id),
        FOREIGN KEY(slot_id) REFERENCES slots(slot_id),
        UNIQUE(user_id, test_id, slot_id)
    )
    """)

    # Seed tests if table is empty
    cursor.execute("SELECT COUNT(*) FROM tests")
    tests_count = cursor.fetchone()[0]

    if tests_count == 0:
        tests_data = [
            ("CBC", "Blood", 350.0),
            ("Blood Sugar", "Blood", 200.0),
            ("Lipid Profile", "Blood", 800.0),
            ("Liver Function Test", "Blood", 900.0),
            ("Thyroid Test", "Hormone", 600.0),
            ("Vitamin D", "Vitamin", 1200.0),
            ("Vitamin B12", "Vitamin", 700.0),
            ("HbA1c", "Blood", 500.0)
        ]
        cursor.executemany("""
            INSERT OR IGNORE INTO tests (test_name, category, price)
            VALUES (?, ?, ?)
        """, tests_data)

    # Seed slots if table is empty
    cursor.execute("SELECT COUNT(*) FROM slots")
    slots_count = cursor.fetchone()[0]

    if slots_count == 0:
        cursor.execute("SELECT test_id FROM tests ORDER BY test_id")
        test_ids = [row[0] for row in cursor.fetchall()]

        for test_id in test_ids:
            for i in range(3):
                cursor.execute("""
                    INSERT OR IGNORE INTO slots (test_id, slot_time, is_booked)
                    VALUES (?, ?, 0)
                """, (test_id, f"2026-11-{10+i:02d} 09:00"))

        # Make test 8 (HbA1c) have no free slots to test slot conflict handling
        cursor.execute("""
            UPDATE slots SET is_booked = 1
            WHERE test_id = ?
        """, (test_ids[-1],))

    conn.commit()
    conn.close()

def get_tests_df() -> pd.DataFrame:
    """Retrieve all diagnostic tests as DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT test_id, test_name, category, price FROM tests ORDER BY test_id", conn)
    conn.close()
    return df

def get_slots_df(test_id: int | None = None) -> pd.DataFrame:
    """Retrieve appointment slots, optionally filtered by test_id."""
    conn = get_connection()
    if test_id is not None:
        query = """
            SELECT s.slot_id, s.test_id, t.test_name, s.slot_time, s.is_booked
            FROM slots s
            JOIN tests t ON s.test_id = t.test_id
            WHERE s.test_id = ?
            ORDER BY s.slot_time
        """
        df = pd.read_sql_query(query, conn, params=(test_id,))
    else:
        query = """
            SELECT s.slot_id, s.test_id, t.test_name, s.slot_time, s.is_booked
            FROM slots s
            JOIN tests t ON s.test_id = t.test_id
            ORDER BY s.test_id, s.slot_time
        """
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_bookings_df(user_id: str | None = None) -> pd.DataFrame:
    """Retrieve bookings, optionally filtered by user_id."""
    conn = get_connection()
    if user_id:
        query = """
            SELECT b.booking_id, b.user_id, t.test_name, t.price, s.slot_time, b.status
            FROM bookings b
            JOIN tests t ON b.test_id = t.test_id
            JOIN slots s ON b.slot_id = s.slot_id
            WHERE b.user_id = ?
            ORDER BY b.booking_id DESC
        """
        df = pd.read_sql_query(query, conn, params=(user_id,))
    else:
        query = """
            SELECT b.booking_id, b.user_id, t.test_name, t.price, s.slot_time, b.status
            FROM bookings b
            JOIN tests t ON b.test_id = t.test_id
            JOIN slots s ON b.slot_id = s.slot_id
            ORDER BY b.booking_id DESC
        """
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df
