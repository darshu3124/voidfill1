import sqlite3
import os

def migrate_db():
    db_path = os.path.join('instance', 'omr_system.db')
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Add exam_name to result table
    c.execute("PRAGMA table_info(result)")
    columns = [col[1] for col in c.fetchall()]
    if 'exam_name' not in columns:
        c.execute("ALTER TABLE result ADD COLUMN exam_name VARCHAR(100) DEFAULT 'Exam 1'")
        print("Added exam_name to result table.")
        
    # Recreate answer_key table to fix unique constraint and add exam_name
    c.execute("PRAGMA table_info(answer_key)")
    ak_columns = [col[1] for col in c.fetchall()]
    if 'exam_name' not in ak_columns:
        c.execute("""
            CREATE TABLE new_answer_key (
                id INTEGER NOT NULL PRIMARY KEY,
                subject_id INTEGER NOT NULL,
                exam_name VARCHAR(100) NOT NULL DEFAULT 'Exam 1',
                question_number INTEGER NOT NULL,
                correct_option VARCHAR(1) NOT NULL,
                FOREIGN KEY(subject_id) REFERENCES subject (id),
                UNIQUE (subject_id, exam_name, question_number)
            )
        """)
        c.execute("""
            INSERT INTO new_answer_key (id, subject_id, question_number, correct_option)
            SELECT id, subject_id, question_number, correct_option FROM answer_key
        """)
        c.execute("DROP TABLE answer_key")
        c.execute("ALTER TABLE new_answer_key RENAME TO answer_key")
        print("Recreated answer_key table successfully with exam_name.")
        
    conn.commit()
    conn.close()
    print("Database migration complete.")

if __name__ == '__main__':
    migrate_db()
