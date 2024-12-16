

import time
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import streamlit as st
from PIL import Image
import pandas as pd

# Function to load known faces and names
def load_known_faces():
    known_faces = []  # List to store known face encodings
    known_names = []  # List to store names corresponding to face encodings
    known_faces_folder = "known_faces"
    if not os.path.exists(known_faces_folder):
        st.error(f"Folder '{known_faces_folder}' does not exist. Please add some known face images.")
        return known_faces, known_names
    
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure you're only processing image files
            img = cv2.imread(f"{known_faces_folder}/{filename}")
            if img is None:
                continue  # Skip any unreadable files
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                known_faces.append(face_encodings[0])  # Only take the first face encoding
                known_names.append(filename.split('.')[0])  # Use filename without extension as name
    return known_faces, known_names

# Function to mark attendance (with punch in/out)
def mark_attendance(name, action="entry"):
    # Check if the attendance file exists, if not create it with header
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Serial Number,Name,Time,Action\n")  # Write header with action column for punch-in/out

    # Try reading the attendance file
    try:
        df = pd.read_csv("attendance.csv")
        # If the dataframe is empty, initialize it with the headers (and serial number 1)
        if df.empty:
            serial_number = 1
        else:
            serial_number = len(df) + 1  # Serial number is one more than the current number of entries
    except pd.errors.EmptyDataError:  # Handle case where the file is empty
        serial_number = 1  # If the file is empty, start with serial number 1
    except FileNotFoundError:
        serial_number = 1  # If the file doesn't exist, start with serial number 1

    # Append attendance entry with the current time and serial number
    with open("attendance.csv", "a") as f:
        f.write(f"{serial_number},{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{action}\n")
    st.success(f"Attendance {action} successfully marked for {name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Function to recognize multiple faces in a frame
def recognize_multiple_faces(frame, rgb_frame, known_faces, known_names, threshold=0.6):
    # Find face locations and encodings in the image
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        name = "Unknown"
        if matches[best_match_index] and face_distances[best_match_index] < threshold:
            name = known_names[best_match_index]
        
        names.append(name)
    
    # Draw rectangles and labels for each detected face
    for i, (face_location, name) in enumerate(zip(face_locations, names)):
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # If a recognized face is found, ask for entry or exit
        if name != "Unknown":
            # Append index to make the key unique for each person
            unique_key = f"{name}_{i}"
            action = st.selectbox(f"Mark attendance for {name}", ["entry", "exit"], key=unique_key)
            if st.button(f"Mark {name} attendance as {action}", key=f"{unique_key}_button"):
                mark_attendance(name, action)
    
    return frame

# Function to register a new face
def register_face(known_faces, known_names):
    st.subheader("Register a New Face")
    new_face_image = st.file_uploader("Upload a new face image", type=["jpg", "png"])
    new_name = st.text_input("Enter the name of the person")

    if new_face_image and new_name:
        image = Image.open(new_face_image)
        img = np.array(image)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        face_encodings = face_recognition.face_encodings(rgb_img)

        if face_encodings:
            known_faces.append(face_encodings[0])  # Add the new face encoding to the list
            known_names.append(new_name)  # Add the new name to the list
            st.success(f"Face of {new_name} registered successfully!")
        else:
            st.error("No face detected in the uploaded image. Please try again.")
    return known_faces, known_names

# Streamlit setup
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# Add custom CSS to style the app
st.markdown(""" 
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #444444;
        }
        h1, h2, h3 {
            color: #2e3a87;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #2e3a87;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 25px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #4c6bdb;
            transform: scale(1.05);
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            max-width: 100%;
        }
        .stDataFrame {
            font-family: 'Courier New', Courier, monospace;
            border-radius: 10px;
            border: 2px solid #2e3a87;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            padding: 10px;
            background-color: #ffffff;
        }
        .stAlert {
            border-left: 4px solid #4caf50;
            background-color: #f1f8e9;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Home Page
def home_page(): 
    st.title("Welcome to the Face Recognition Attendance System")
    st.subheader("This is a real-time attendance system based on facial recognition technology.")
    
    try:
        img = Image.open('welcome_image.jpg')
        st.image(img, use_column_width=True)
    except FileNotFoundError:
        st.warning("Welcome image not found. Please add 'welcome_image.jpg' in your project directory.")
    
    st.write(""" 
        ### About this Application:
        This system uses **face recognition** to automatically mark attendance based on the faces it detects via your webcam.
        
        ### Features:
        - **Real-time attendance** marking through face recognition.
        - **CSV log** of all attendance records.
        - **Webcam feed control**: Start/Stop webcam feed to begin recognition.
        
        ### How to Use:
        1. Add images of known individuals to the **'known_faces'** folder.
        2. Register new faces if needed.
        3. Start the webcam feed and face recognition will automatically mark attendance.

        **Note:** Ensure your webcam is properly connected for real-time recognition.
    """)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Mark Attendance", "Register New Face", "View Attendance"])

# Load known faces and names
known_faces, known_names = load_known_faces()

# Handle different pages
if page == "Home":
    home_page()
elif page == "Mark Attendance":
    st.subheader("Mark Attendance")
    
    # Handle session state for webcam feed
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    start_webcam = st.checkbox("Start Webcam", value=st.session_state.webcam_running)

    if start_webcam != st.session_state.webcam_running:
        st.session_state.webcam_running = start_webcam

    # Start webcam feed only if checkbox is checked
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not access the webcam.")
        else:
            stframe = st.empty()
            frame_counter = 0
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process every 5th frame for efficiency
                if frame_counter % 5 == 0:
                    frame = recognize_multiple_faces(frame, rgb_frame, known_faces, known_names)
                frame_counter += 1

                stframe.image(frame, channels="BGR", use_column_width=True)
                time.sleep(1)
            cap.release()
    else:
        st.warning("Webcam is off. Please start the webcam to begin recognition.")
elif page == "Register New Face":
    known_faces, known_names = register_face(known_faces, known_names)
elif page == "View Attendance":
    st.subheader("Attendance Records")
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("No attendance records found.")








