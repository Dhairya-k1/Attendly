# 🎓 Attendly: Zero-Touch Biometric Attendance System

Attendly is a highly novel, hardware-free attendance tracking platform. It upgrades standard facial recognition by utilizing a **BYOD (Bring Your Own Device)** architecture paired with **Ultrasonic Proof-of-Presence** and **Active Client-Side Liveness Detection** to mathematically prevent proxy attendance and photographic spoofing.

## ✨ Core Innovations

* **Ultrasonic Acoustic Tokening:**
  Drops reliance on easily spoofed GPS data. The professor's dashboard emits an inaudible 19kHz sine wave containing a rolling session token. A student's device *must* physically hear this acoustic token via its microphone to unlock the check-in button, guaranteeing they are physically present in the room.

* **Active Liveness Detection (Edge ML):**
  Defeats 2D printed photograph attacks. The student's device runs lightweight machine learning in the browser to detect facial muscle movements (a "Smile Challenge"). The camera will not unlock until liveness is verified >80% confidence locally on the edge device.

* **BYOD Architecture:**
  Eliminates the CapEx of installing centralized cameras in every classroom. The system leverages the smartphones students already own via a mobile-responsive web app.

* **FastAPI Microservice:**
  Decouples the heavy facial recognition matching logic from the frontend, ensuring high server throughput and scalability.

## 🛠️ Technology Stack

* **Backend:** Python 3, FastAPI, Uvicorn
* **Backend Computer Vision:** OpenCV, `face_recognition` (dlib)
* **Frontend Edge ML (Liveness):** `face-api.js` (TensorFlow.js core)
* **Audio Engineering:** Web Audio API (FFT for 19kHz detection)
* **Database:** SQLite3
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## 📁 Project Structure

```
/Attendly-Project/
│
├── main.py
├── register_user.py
├── encode_faces.py
├── requirements.txt
│
├── /models/
│   ├── tiny_face_detector_model-weights_manifest.json
│   ├── tiny_face_detector_model-shard1
│   ├── face_expression_model-weights_manifest.json
│   └── face_expression_model-shard1
│
├── /dataset/
├── encodings.pickle
├── attendance_system.db
│
└── /templates/
    ├── index.html
    ├── professor.html
    └── student.html
```

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/Attendly.git
cd Attendly
```

### 2. Set up Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Backend Dependencies

> ⚠️ Installing `dlib` requires C++ Build Tools

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy face_recognition
python -m pip install git+https://github.com/ageitgey/face_recognition_models
```

### 4. Download Frontend ML Models

Download required `face-api.js` weights and place them in:

```
/models/
```

---

## 💻 Usage Instructions

### Phase 1: Registration

```bash
python register_user.py
```

* Enter student name & role
* Capture 5 face images

Then run:

```bash
python encode_faces.py
```

---

### Phase 2: Run the System

#### Start Backend Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Open Platform

```
http://localhost:8000
```

#### Professor Workflow

* Click **Professor Login**
* Click **Start Class & Emit Token**
* System emits inaudible 19kHz tone

#### Student Workflow

Connect via:

```
http://192.168.X.X:8000/app
```

**Security Locks:**

1. 🎤 Microphone detects 19kHz tone
2. 😊 Smile detection (liveness check)

Once verified:

* Capture selfie
* Authenticate with backend


