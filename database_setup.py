import sqlite3
import os

def initialize_database():
    """
    Creates the SQLite database and necessary tables for the Smart Attendance System.
    """
    db_name = "attendance_system.db"
    
    # Check if database already exists
    if os.path.exists(db_name):
        print(f"Database '{db_name}' already exists. Connecting to it...")
    else:
        print(f"Creating new database '{db_name}'...")

    # Connect to the SQLite database (this creates the file if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create USERS Table
    # Stores the basic information of the person to be recognized
    # Note: We don't store the actual image here, but we will store the face encodings 
    # as a blob/text later, or save images in a folder and link by ID.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'Student',
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("- 'Users' table ready.")

    # Create ATTENDANCE Table
    # Logs every time a user is recognized by the camera
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

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    print("Database initialization complete!")

if __name__ == "__main__":
    initialize_database()