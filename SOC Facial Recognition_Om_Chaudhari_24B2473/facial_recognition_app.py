import cv2
import os
import numpy as np
from datetime import datetime

class FacialRecognitionApp:
    def __init__(self):
        # Initialize the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Directory to store known faces
        self.known_faces_dir = "known_faces"
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        
        # Dictionary to store known face encodings and names
        self.known_faces = {}
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from the known_faces directory"""
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                face_images = []
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            face_images.append(img)
                
                if face_images:
                    self.known_faces[person_name] = face_images
                    print(f"Loaded {len(face_images)} images for {person_name}")
    
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces, gray
    
    def recognize_face(self, face_roi):
        """Simple face recognition using template matching"""
        best_match = "Unknown"
        best_score = float('inf')
        
        for person_name, face_images in self.known_faces.items():
            for known_face in face_images:
                # Resize the face ROI to match the known face size
                resized_face = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))
                
                # Calculate the difference between faces
                diff = cv2.absdiff(resized_face, known_face)
                score = np.sum(diff)
                
                if score < best_score:
                    best_score = score
                    best_match = person_name
        
        # Set a threshold for recognition
        threshold = 1000000  # Adjust this value based on testing
        if best_score > threshold:
            best_match = "Unknown"
        
        return best_match, best_score
    
    def collect_data(self, person_name, num_images=30):
        """Collect face data for a person"""
        person_dir = os.path.join(self.known_faces_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"Collecting {num_images} images for {person_name}")
        print("Press 'c' to capture an image, 'q' to quit")
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            faces, gray = self.detect_faces(frame)
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Press 'c' to capture ({count}/{num_images})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                # Save the first detected face
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                img_name = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                cv2.imwrite(img_name, face_roi)
                count += 1
                print(f"Captured image {count}/{num_images}")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Reload known faces
        self.load_known_faces()
        print(f"Data collection complete for {person_name}")
    
    def real_time_recognition(self):
        """Perform real-time face recognition"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time face recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            faces, gray = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize the face
                name, score = self.recognize_face(face_roi)
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({score:.0f})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow('Facial Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    app = FacialRecognitionApp()
    
    while True:
        print("\n=== Facial Recognition App ===")
        print("1. Collect face data")
        print("2. Start real-time recognition")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            person_name = input("Enter person's name: ")
            num_images = int(input("Enter number of images to collect (default 30): ") or 30)
            app.collect_data(person_name, num_images)
        elif choice == '2':
            app.real_time_recognition()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

