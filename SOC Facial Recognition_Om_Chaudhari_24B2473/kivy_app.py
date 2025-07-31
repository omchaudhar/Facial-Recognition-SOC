from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import cv2
import threading
import os

class FacialRecognitionGUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10
        
        # Initialize camera
        self.capture = None
        self.is_capturing = False
        
        # Face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Known faces directory
        self.known_faces_dir = "known_faces"
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title = Label(text='Facial Recognition App', size_hint_y=None, height=50, 
                     font_size=24, bold=True)
        self.add_widget(title)
        
        # Camera display
        self.camera_image = Image(size_hint_y=0.6)
        self.add_widget(self.camera_image)
        
        # Status label
        self.status_label = Label(text='Ready', size_hint_y=None, height=30)
        self.add_widget(self.status_label)
        
        # Input for person name
        name_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        name_layout.add_widget(Label(text='Person Name:', size_hint_x=0.3))
        self.name_input = TextInput(multiline=False, size_hint_x=0.7)
        name_layout.add_widget(self.name_input)
        self.add_widget(name_layout)
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        self.start_camera_btn = Button(text='Start Camera')
        self.start_camera_btn.bind(on_press=self.start_camera)
        button_layout.add_widget(self.start_camera_btn)
        
        self.collect_data_btn = Button(text='Collect Data')
        self.collect_data_btn.bind(on_press=self.collect_data)
        button_layout.add_widget(self.collect_data_btn)
        
        self.recognize_btn = Button(text='Start Recognition')
        self.recognize_btn.bind(on_press=self.start_recognition)
        button_layout.add_widget(self.recognize_btn)
        
        self.stop_btn = Button(text='Stop')
        self.stop_btn.bind(on_press=self.stop_camera)
        button_layout.add_widget(self.stop_btn)
        
        self.add_widget(button_layout)
    
    def start_camera(self, instance):
        if not self.is_capturing:
            self.capture = cv2.VideoCapture(0)
            self.is_capturing = True
            self.status_label.text = 'Camera started'
            Clock.schedule_interval(self.update_camera, 1.0/30.0)  # 30 FPS
    
    def stop_camera(self, instance):
        if self.is_capturing:
            self.is_capturing = False
            Clock.unschedule(self.update_camera)
            if self.capture:
                self.capture.release()
            self.status_label.text = 'Camera stopped'
            # Clear the camera display
            self.camera_image.texture = None
    
    def update_camera(self, dt):
        if self.capture and self.is_capturing:
            ret, frame = self.capture.read()
            if ret:
                # Convert frame to texture and display
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.camera_image.texture = texture
    
    def collect_data(self, instance):
        person_name = self.name_input.text.strip()
        if not person_name:
            self.show_popup('Error', 'Please enter a person name')
            return
        
        if not self.is_capturing:
            self.show_popup('Error', 'Please start the camera first')
            return
        
        # Create directory for person
        person_dir = os.path.join(self.known_faces_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        self.status_label.text = f'Collecting data for {person_name}...'
        
        # Collect 30 images
        threading.Thread(target=self.collect_images_thread, args=(person_name,)).start()
    
    def collect_images_thread(self, person_name):
        person_dir = os.path.join(self.known_faces_dir, person_name)
        count = 0
        max_images = 30
        
        while count < max_images and self.is_capturing:
            if self.capture:
                ret, frame = self.capture.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        # Save the first detected face
                        x, y, w, h = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        img_name = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                        cv2.imwrite(img_name, face_roi)
                        count += 1
                        
                        # Update status on main thread
                        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 
                                                             f'Collected {count}/{max_images} images'), 0)
            
            # Wait a bit between captures
            threading.Event().wait(0.5)
        
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 
                                             f'Data collection complete for {person_name}'), 0)
    
    def start_recognition(self, instance):
        if not self.is_capturing:
            self.show_popup('Error', 'Please start the camera first')
            return
        
        self.status_label.text = 'Recognition mode active'
        # In a real implementation, you would start the recognition process here
        # For now, we'll just update the status
    
    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(0.6, 0.4))
        popup.open()

class FacialRecognitionApp(App):
    def build(self):
        return FacialRecognitionGUI()

if __name__ == '__main__':
    FacialRecognitionApp().run()

