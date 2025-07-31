import cv2
import os
import time

# Path to store collected images
DATA_PATH = os.path.join('data')

# Number of images to collect for each class
number_images = 50

# Prompt for the name of the person to collect data for
person_name = input('Enter the name of the person for data collection: ')

# Create a directory for the person if it doesn't exist
person_path = os.path.join(DATA_PATH, person_name)
os.makedirs(person_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

print(f"Collecting {number_images} images for {person_name}. Press 'q' to quit.")

for imgnum in range(number_images):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    imgname = os.path.join(person_path, f'{person_name}_{imgnum}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('Image Collection', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
print('Data collection complete.')


