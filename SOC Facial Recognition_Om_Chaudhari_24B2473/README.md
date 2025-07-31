# Facial Recognition Application

A Python-based facial recognition application built with OpenCV and Kivy, inspired by the YouTube tutorial series "Build a Facial Recognition App // Deep Learning Project // Paper2Code Series" by Nicholas Renotte.

## Features

- Real-time face detection using OpenCV's Haar Cascades
- Data collection for training custom face recognition
- Simple face recognition using template matching
- Graphical user interface built with Kivy
- Command-line interface for basic operations

## Requirements

- Python 3.7 or higher
- Webcam (for real-time face detection and data collection)
- Operating System: Windows, macOS, or Linux

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Run the main application:
```bash
python facial_recognition_app.py
```

This will present you with a menu:
1. Collect face data - Capture images for training
2. Start real-time recognition - Begin face recognition
3. Exit

### Kivy GUI Application

Run the GUI version:
```bash
python kivy_app.py
```

The GUI provides:
- Camera preview
- Data collection interface
- Real-time recognition controls

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in real-time
2. **Data Collection**: Captures and stores face images for each person
3. **Face Recognition**: Uses template matching to compare detected faces with stored data
4. **User Interface**: Provides both command-line and graphical interfaces

## File Structure

```
facial-recognition-app/
├── facial_recognition_app.py  # Main command-line application
├── kivy_app.py               # GUI application
├── collect_data.py           # Data collection script
├── preprocess_data.py        # Data preprocessing utilities
├── train_model.py            # Model training (placeholder)
├── test_app.py              # Test suite
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── known_faces/             # Directory for stored face data
    └── [person_name]/       # Individual person directories
        └── *.jpg            # Face images
```

## Limitations

- This implementation uses basic template matching for face recognition, which is less accurate than deep learning approaches
- Camera access may not work in all environments (especially virtual machines or containers)
- Recognition accuracy depends on lighting conditions and face angles
- No advanced features like face verification or anti-spoofing

## Future Improvements

- Implement deep learning-based face recognition (Siamese Neural Networks)
- Add face verification capabilities
- Improve recognition accuracy with better algorithms
- Add database storage for face encodings
- Implement real-time performance optimizations

## Troubleshooting

### Camera Issues
- Ensure your webcam is connected and not being used by other applications
- Try different camera indices (0, 1, 2) if the default doesn't work
- Check camera permissions on your operating system

### Installation Issues
- Make sure you have the latest version of pip: `pip install --upgrade pip`
- On Linux, you may need to install additional system packages for OpenCV
- For Kivy issues, refer to the official Kivy installation guide

## Credits

This project is based on the YouTube tutorial series by Nicholas Renotte:
"Build a Facial Recognition App // Deep Learning Project // Paper2Code Series"

## License

This project is for educational purposes. Please respect privacy and obtain consent before collecting face data.

