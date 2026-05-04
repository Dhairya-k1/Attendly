import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import face_recognition
import pickle
import numpy as np
import sqlite3
from datetime import datetime

# --- Configuration ---
ENCODING_FILE = "encodings.pickle"
DB_NAME = "attendance_system.db"

# Initialize FastAPI App
app = FastAPI(title="Attendly Platform")

app.mount("/models", StaticFiles(directory="models"), name="models")

# --- WEB ROUTING (THE SINGLE PLATFORM LOGIC) ---

def serve_html(file_name):
    """Helper function to load HTML files from the templates folder"""
    file_path = os.path.join("templates", file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content=f"<h1>Error: {file_name} not found</h1>", status_code=404)

@app.get("/")
def read_root():
    """Serves the Unified Landing Page"""
    return serve_html("index.html")

@app.get("/professor")
def professor_dashboard():
    """Serves the Professor's Audio Emitter Dashboard"""
    return serve_html("professor.html")

@app.get("/app")
def student_app():
    """Serves the Student's BYOD Web App"""
    return serve_html("student.html")

# Allow requests from any frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Encodings into memory when the server starts
try:
    with open(ENCODING_FILE, "rb") as f:
        known_data = pickle.load(f)
    print(f"[STARTUP] Successfully loaded {len(known_data['names'])} face encodings.")
except FileNotFoundError:
    print(f"[ERROR] {ENCODING_FILE} not found. Please run encode_faces.py first.")
    known_data = None

# In-memory cache to prevent database spam (Debounce)
local_attendance_cache = {}

def log_attendance_to_db(user_id, name):
    """Logs the user to SQLite if they haven't been logged today."""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    if local_attendance_cache.get(user_id) == current_date:
        return "Already logged today."
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM Attendance WHERE user_id=? AND date=?", (user_id, current_date))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO Attendance (user_id, date, time) VALUES (?, ?, ?)", 
                       (user_id, current_date, current_time))
        conn.commit()
        local_attendance_cache[user_id] = current_date
        conn.close()
        return "Attendance marked successfully."
    
    conn.close()
    return "Already logged today."


@app.get("/")
def read_root():
    """Health check endpoint to ensure the server is running."""
    return {"status": "online", "message": "Attendly API is running."}


@app.post("/mark_attendance")
async def mark_attendance(
    audio_token: str = Form(...), 
    image: UploadFile = File(...)
):
    """
    The core endpoint. It receives an audio token and a selfie from the student's phone.
    """
    if known_data is None:
        raise HTTPException(status_code=500, detail="Server misconfigured: No face encodings loaded.")

    # 1. VERIFY THE ULTRASONIC TOKEN
    # In the future, this will check against a live database of active class sessions.
    # For now, we simulate a valid hardcoded token.
    valid_active_token = "CS101-May03" 
    
    if audio_token != valid_active_token:
        raise HTTPException(status_code=403, detail="Authentication Failed: Invalid or expired audio token. Are you actually in the classroom?")

    # 2. PROCESS THE UPLOADED IMAGE
    try:
        # Read the file uploaded by the phone into memory
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Smartphone photos are massive (12MP+). We MUST resize it or the server will crash/hang.
        # Resize width to 800px, maintaining aspect ratio
        ratio = 800.0 / frame.shape[1]
        dim = (800, int(frame.shape[0] * ratio))
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB for the face_recognition library
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image file: {str(e)}")

    # 3. RUN FACIAL RECOGNITION
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return {"status": "failed", "message": "No face detected in the image."}
    if len(face_locations) > 1:
        return {"status": "failed", "message": "Multiple faces detected. Please take a selfie alone."}

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    uploaded_encoding = face_encodings[0]

    # Compare against our known dataset
    matches = face_recognition.compare_faces(known_data["encodings"], uploaded_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_data["encodings"], uploaded_encoding)
    
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_data["names"][best_match_index]
            user_id = known_data["ids"][best_match_index]
            
            # 4. LOG ATTENDANCE
            db_result = log_attendance_to_db(user_id, name)
            
            return {
                "status": "success", 
                "name": name, 
                "message": db_result,
                "token_verified": True
            }

    return {"status": "failed", "message": "Face not recognized. You are not registered in the system."}

@app.get("/get_attendance")
def get_attendance():
    """Fetches today's attendance list for the professor's dashboard."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Get the names and times of everyone who checked in today
    cursor.execute('''
        SELECT Users.name, Attendance.time 
        FROM Attendance 
        JOIN Users ON Attendance.user_id = Users.user_id
        WHERE Attendance.date = ?
        ORDER BY Attendance.time DESC
    ''', (current_date,))
    
    records = cursor.fetchall()
    conn.close()
    
    # Format the data into a list of dictionaries for the frontend
    return [{"name": row[0], "time": row[1]} for row in records]

@app.get("/app", response_class=HTMLResponse)
def get_student_app():
    """Serves the student HTML app to mobile devices."""
    try:
        with open("student_app.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "student_app.html not found on server."