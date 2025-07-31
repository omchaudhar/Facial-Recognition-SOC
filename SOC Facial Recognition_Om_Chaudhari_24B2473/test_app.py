import cv2
import os
import sys

def test_opencv():
    """Test OpenCV installation and face detection"""
    print("Testing OpenCV...")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test face cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Face cascade path: {face_cascade_path}")
    print(f"Face cascade exists: {os.path.exists(face_cascade_path)}")
    
    # Try to load the cascade
    try:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        print("Face cascade loaded successfully")
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        return False
    
    return True

def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Camera opened successfully")
            ret, frame = cap.read()
            if ret:
                print(f"Frame captured successfully. Shape: {frame.shape}")
            else:
                print("Failed to capture frame")
            cap.release()
            return True
        else:
            print("Failed to open camera")
            return False
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return False

def test_directories():
    """Test directory creation"""
    print("\nTesting directory creation...")
    test_dir = "test_known_faces"
    try:
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        print(f"Directory '{test_dir}' created/exists")
        
        # Clean up
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
        print("Test directory cleaned up")
        return True
    except Exception as e:
        print(f"Error with directory operations: {e}")
        return False

def test_image_processing():
    """Test basic image processing"""
    print("\nTesting image processing...")
    try:
        # Create a dummy image
        import numpy as np
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test color conversion
        gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
        print(f"Color conversion successful. Gray shape: {gray.shape}")
        
        # Test resize
        resized = cv2.resize(gray, (50, 50))
        print(f"Resize successful. New shape: {resized.shape}")
        
        return True
    except Exception as e:
        print(f"Error in image processing: {e}")
        return False

def main():
    print("=== Facial Recognition App Test Suite ===\n")
    
    tests = [
        ("OpenCV", test_opencv),
        ("Camera", test_camera),
        ("Directories", test_directories),
        ("Image Processing", test_image_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Unexpected error in {test_name} test: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed! The application should work correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

