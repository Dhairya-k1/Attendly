import cv2
import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
ENCODING_FILE = "encodings.pickle"

def encode_known_faces():
    print("[INFO] Quantifying faces...")
    known_encodings = []
    known_names = []
    known_ids = []

    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory '{DATASET_DIR}' not found. Please run register_user.py first.")
        return

    # Loop over the user folders in the dataset directory
    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        # The folder name is formatted as "ID_Name", let's split it
        try:
            user_id, user_name = folder_name.split('_', 1)
        except ValueError:
            print(f"[WARNING] Skipping folder {folder_name} (Incorrect naming format)")
            continue

        print(f"[INFO] Processing images for {user_name}...")

        # Loop over the images inside the user's folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            # OpenCV loads images in BGR (Blue, Green, Red) format, 
            # but face_recognition needs RGB (Red, Green, Blue).
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect the (x, y) coordinates of the bounding boxes corresponding to each face
            boxes = face_recognition.face_locations(rgb_image, model="hog")

            # Compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb_image, boxes)

            # Loop over the encodings (there should ideally be just one per image)
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(user_name)
                known_ids.append(user_id)

    # Dump the facial encodings + names to disk
    print("[INFO] Serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names, "ids": known_ids}
    
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print(f"[SUCCESS] Saved encodings to {ENCODING_FILE}")

if __name__ == "__main__":
    encode_known_faces()