import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import csv
import os

# --- Configuration ---
ENCODING_FILE = "encodings.pickle"
DB_NAME = "attendance_system.db"

class AttendanceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x600")
        self.window.configure(bg="#f0f0f0")

        self.local_attendance_cache = {}
        
        # Load Encodings
        self.known_data = self.load_encodings()

        # --- GUI Layout ---
        # Left Frame: Video Feed
        self.video_frame = tk.Frame(self.window, bg="black", width=640, height=480)
        self.video_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()

        # Right Frame: Controls and Logs
        self.control_frame = tk.Frame(self.window, bg="#f0f0f0")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.title_label = tk.Label(self.control_frame, text="Smart Attendance", font=("Helvetica", 20, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=(0, 20))

        self.export_btn = tk.Button(self.control_frame, text="Export to CSV", font=("Helvetica", 12), bg="#4CAF50", fg="white", command=self.export_to_csv)
        self.export_btn.pack(fill=tk.X, pady=5)

        self.quit_btn = tk.Button(self.control_frame, text="Quit Application", font=("Helvetica", 12), bg="#f44336", fg="white", command=self.on_closing)
        self.quit_btn.pack(fill=tk.X, pady=5)

        tk.Label(self.control_frame, text="Recent Scans:", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=(20, 5), anchor="w")
        
        # Listbox for live attendance logs
        self.log_list = tk.Listbox(self.control_frame, font=("Helvetica", 11), height=15)
        self.log_list.pack(fill=tk.BOTH, expand=True)

        # --- Video Capture Setup ---
        self.vid = cv2.VideoCapture(0)
        
        # Handle Window Close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the video loop
        if self.known_data:
            self.delay = 15 # milliseconds
            self.update_frame()
        else:
            messagebox.showerror("Error", "Could not load encodings. Please run encode_faces.py first.")

    def load_encodings(self):
        try:
            with open(ENCODING_FILE, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def mark_attendance(self, user_id, name):
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
        if self.local_attendance_cache.get(user_id) == current_date:
            return 
            
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM Attendance WHERE user_id=? AND date=?", (user_id, current_date))
        if cursor.fetchone() is None:
            cursor.execute("INSERT INTO Attendance (user_id, date, time) VALUES (?, ?, ?)", (user_id, current_date, current_time))
            conn.commit()
            
            # Update GUI Listbox
            log_msg = f"[{current_time}] {name} - Present"
            self.log_list.insert(0, log_msg) # Insert at top
            
        self.local_attendance_cache[user_id] = current_date
        conn.close()

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            # 1. Process Face Recognition (Scaled down for speed)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_data["encodings"], face_encoding, tolerance=0.5)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_data["encodings"], face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_data["names"][best_match_index]
                        user_id = self.known_data["ids"][best_match_index]
                        self.mark_attendance(user_id, name)

                # Scale coordinates back up and draw boxes
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 2. Convert Frame for Tkinter
            # OpenCV uses BGR, Pillow uses RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # 3. Schedule next frame update (This replaces the 'while True' loop)
        self.window.after(self.delay, self.update_frame)

    def export_to_csv(self):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT Attendance.date, Attendance.time, Users.name, Users.role 
            FROM Attendance JOIN Users ON Attendance.user_id = Users.user_id
            ORDER BY Attendance.date DESC, Attendance.time DESC
        ''')
        records = cursor.fetchall()
        
        if not records:
            messagebox.showinfo("Export", "No attendance records found.")
            return

        filename = f"Attendance_Report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Time', 'Name', 'Role'])
            writer.writerows(records)
            
        messagebox.showinfo("Success", f"Exported {len(records)} records to {filename}")
        conn.close()

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
  
    root = tk.Tk()
    app = AttendanceApp(root, "Smart Face Recognition Attendance")
    root.mainloop()