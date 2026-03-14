import cv2
import face_recognition
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import csv

# --- Configuration ---
ENCODING_FILE = "encodings.pickle"
DB_NAME = "attendance_system.db"

# Dictionary to keep track of who was marked present today to prevent database spam
# Format will be: {'user_id': 'YYYY-MM-DD'}
local_attendance_cache = {}

def mark_attendance(user_id, name):
    """Logs the user's attendance into the SQLite database if not already logged today."""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    # 1. Quick Local Check (Debounce)
    if local_attendance_cache.get(user_id) == current_date:
        return # Already marked present during this session, do nothing.
        
    # 2. Database Check (In case the script was restarted today)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM Attendance WHERE user_id=? AND date=?", (user_id, current_date))
    record = cursor.fetchone()
    
    if record is None:
        # 3. Log the Attendance
        cursor.execute("INSERT INTO Attendance (user_id, date, time) VALUES (?, ?, ?)", 
                       (user_id, current_date, current_time))
        conn.commit()
        print(f"\n[ATTENDANCE] Success! Marked {name} present at {current_time}.")
        
    # Update local cache so we don't query the database again for this person today
    local_attendance_cache[user_id] = current_date
    conn.close()

def export_to_csv():
    """Generates a CSV report of all attendance records."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # SQL Query to join the Users and Attendance tables to get readable names
    cursor.execute('''
        SELECT Attendance.date, Attendance.time, Users.name, Users.role 
        FROM Attendance 
        JOIN Users ON Attendance.user_id = Users.user_id
        ORDER BY Attendance.date DESC, Attendance.time DESC
    ''')
    records = cursor.fetchall()
    
    if not records:
        print("\n[EXPORT] No attendance records found to export.")
        return

    # Create a filename with today's date
    filename = f"Attendance_Report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Time', 'Name', 'Role']) # Write Headers
        writer.writerows(records) # Write Data
        
    print(f"\n[EXPORT] Successfully exported {len(records)} records to {filename}")
    conn.close()

def run_system():
    print("[INFO] Loading facial encodings...")
    try:
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] {ENCODING_FILE} not found. Run encode_faces.py first.")
        return

    print("[INFO] Starting video stream...")
    print("[INSTRUCTION] Press 'q' to quit.")
    print("[INSTRUCTION] Press 'e' to export attendance to CSV.")
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
            name = "Unknown"
            user_id = None

            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = data["names"][best_match_index]
                    user_id = data["ids"][best_match_index]
                    
                    # TRIGGER ATTENDANCE LOGGING
                    mark_attendance(user_id, name)

            face_names.append(name)

        # Draw boxes and names on the screen
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Smart Attendance System', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            export_to_csv()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()