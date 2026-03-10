import sqlite3

def upgrade_db():
    conn = sqlite3.connect('instance/omr_system.db')
    c = conn.cursor()

    try:
        c.execute("ALTER TABLE student ADD COLUMN created_at DATETIME")
        print("Added created_at column to student")
    except Exception as e:
        print("student.created_at:", e)

    conn.commit()
    conn.close()
    
if __name__ == '__main__':
    upgrade_db()
