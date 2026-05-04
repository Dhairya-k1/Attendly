import sqlite3
import os

def initialize_database():

    db_name = "attendance_system.db"
 
    if os.path.exists(db_name):
        print(f"Database '{db_name}' already exists. Connecting to it...")
    else:
        print(f"Creating new database '{db_name}'...")

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll TEXT DEFAULT 'Student',
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("- 'Users' table ready.")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES Users (user_id)
        )
    ''')
    print("- 'Attendance' table ready.")

    conn.commit()
    conn.close()
    
    print("Database initialization complete!")

if __name__ == "__main__":
    initialize_database()