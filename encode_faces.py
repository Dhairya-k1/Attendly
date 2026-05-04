import cv2
import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
ENCODING_FILE = "encodings.pickle"

def encode_known_faces():
    print("Quantifying faces...")
    known_encodings = []
    known_names = []
    known_ids = []

    if not os.path.exists(DATASET_DIR):
        print(f"Directory '{DATASET_DIR}' not found. Please run register_user.py first.")
        return

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        try:
            user_id, user_name = folder_name.split('_', 1)
        except ValueError:
            print(f"Skipping folder {folder_name} (Incorrect naming format)")
            continue

        print(f"Processing images for {user_name}...")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb_image, model="hog")

            encodings = face_recognition.face_encodings(rgb_image, boxes)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(user_name)
                known_ids.append(user_id)

    print("Serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names, "ids": known_ids}
    
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Saved encodings to {ENCODING_FILE}")

if __name__ == "__main__":
    encode_known_faces()