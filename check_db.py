import sqlite3
import os

db_path = 'instance/omr_system.db'
if not os.path.exists(db_path):
    # Try alternate location if instance folder is elsewhere
    db_path = 'omr_system.db'

print(f"Checking DB: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT id, name FROM subject WHERE name LIKE '%python%'")
subjects = cursor.fetchall()

if not subjects:
    print("No 'python' subject found.")
else:
    for sub_id, name in subjects:
        print(f"Subject: {name} (ID: {sub_id})")
        cursor.execute("SELECT question_number, correct_option FROM answer_key WHERE subject_id = ?", (sub_id,))
        keys = cursor.fetchall()
        print(f"Found {len(keys)} answer key entries.")
        for q, opt in keys:
            print(f"  Q{q}: {opt}")

conn.close()
