import cv2
import os
import sqlite3
import time

# --- Configuration ---
DATABASE_NAME = "attendance_system.db"
DATASET_DIR = "dataset"

def register_user_in_db(name, role):
    """Inserts a new user into the database and returns their auto-generated ID."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO Users (name, role) VALUES (?, ?)", (name, role))
    user_id = cursor.lastrowid # Get the ID of the newly inserted row
    
    conn.commit()
    conn.close()
    return user_id

def capture_face_images(user_id, user_name, num_images=5):
    """Opens the webcam and captures images of the user's face."""
    # Create the main dataset directory if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # Create a specific folder for this user (e.g., dataset/1_JohnDoe)
    user_folder = os.path.join(DATASET_DIR, f"{user_id}_{user_name.replace(' ', '')}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Initialize the webcam
    print("\n[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0) # 0 is usually the default built-in webcam
    
    # Load OpenCV's pre-trained face detection model to help frame the shot
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    print(f"[INFO] Look at the camera. Capturing {num_images} images...")
    time.sleep(2) # Give the user a brief moment to get ready

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break

        # Convert to grayscale for the face detection algorithm
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Draw a rectangle around the face to guide the user
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {count+1}/{num_images}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Registration", frame)

        # Wait for the user to press 'c' to capture a photo when they are ready
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the raw, un-drawn-on frame (we don't want the green box in our dataset)
            ret_clean, clean_frame = cap.read() 
            img_name = os.path.join(user_folder, f"face_{count+1}.jpg")
            cv2.imwrite(img_name, clean_frame)
            print(f" -> Captured {img_name}")
            count += 1
            time.sleep(0.5) # Brief pause between shots

        # Press 'q' to quit early
        elif key == ord('q'):
            print("[INFO] Registration cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Registration complete for {user_name}! Images saved in {user_folder}")

if __name__ == "__main__":
    print("=== Smart Attendance: User Registration ===")
    name_input = input("Enter user's full name: ")
    role_input = input("Enter user's role (e.g., Student, Employee): ")
    
    print("\n[INFO] Saving to database...")
    new_user_id = register_user_in_db(name_input, role_input)
    
    print(f"[INFO] User saved with ID: {new_user_id}")
    print("[INSTRUCTION] When the camera opens, make sure your face is in the green box.")
    print("[INSTRUCTION] Press the 'c' key on your keyboard to take a photo. Do this 5 times.")
    
    capture_face_images(new_user_id, name_input, num_images=5)