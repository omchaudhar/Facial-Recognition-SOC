import os
import cv2
import numpy as np

DATA_DIR = ".cache/kagglehub/datasets/vasukipatel/face-recognition-dataset/versions/1/Faces/Faces"
PROCESSED_DATA_DIR = "processed_data"

IMG_SIZE = (100, 100) # Target image size

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0 # Normalize pixel values to [0, 1]
    return img

def create_processed_dataset():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    for person_name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person_name)
        if os.path.isdir(person_dir):
            processed_person_dir = os.path.join(PROCESSED_DATA_DIR, person_name)
            os.makedirs(processed_person_dir, exist_ok=True)
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                processed_img = preprocess_image(img_path)
                if processed_img is not None:
                    # Save processed image (optional, can be loaded directly into memory for training)
                    # For now, we'll just process and not save to disk to save space and time
                    pass
    print("Data preprocessing complete.")

if __name__ == "__main__":
    create_processed_dataset()


