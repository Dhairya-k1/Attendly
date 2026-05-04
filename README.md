# 🎓 Attendly: Zero-Touch Biometric Attendance System

Attendly is a highly novel, hardware-free attendance tracking platform. It upgrades standard facial recognition by utilizing a **BYOD (Bring Your Own Device)** architecture paired with **Ultrasonic Proof-of-Presence** to mathematically prevent proxy attendance and GPS spoofing.

## ✨ Core Innovations

*   **Ultrasonic Acoustic Tokening:** Drops reliance on easily spoofed GPS data. The professor's dashboard emits an inaudible 19kHz sine wave containing a rolling session token. A student's device *must* physically hear this acoustic token via its microphone to unlock the check-in button, guaranteeing they are physically present in the room.
*   **BYOD Architecture:** Eliminates the CapEx of installing cameras in every classroom. The system leverages the smartphones students already own via a mobile-responsive web app.
*   **FastAPI Microservice:** Decouples the heavy facial recognition logic from the frontend, processing high-res selfies and database operations centrally.

## 🛠️ Technology Stack

*   **Backend:** Python 3, FastAPI, Uvicorn
*   **Computer Vision:** OpenCV, `face_recognition` (dlib)
*   **Database:** SQLite3
*   **Frontend:** HTML5, CSS3, Vanilla Javascript
*   **Audio Engineering:** Web Audio API (Fast Fourier Transform for 19kHz frequency detection)

## 📁 Project Structure
```text
/Attendly-Project/
│
├── main.py                 # FastAPI backend & core routing
├── register_user.py        # CLI script to capture new user faces
├── encode_faces.py         # Script to generate 128-d embeddings
├── requirements.txt        # Python dependencies
│
├── /dataset/               # Raw captured face images (Git-ignored)
├── encodings.pickle        # Serialized face embeddings (Git-ignored)
├── attendance_system.db    # Live SQLite database (Git-ignored)
│
└── /templates/             
    ├── index.html          # Unified landing page
    ├── professor.html      # Emitter dashboard & live logs
    └── student.html        # Receiver app (Camera + Mic FFT listener)