



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
            f.write("Serial Number,Name,Time,Action \n")  # Write header with action column for punch-in/out

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
            # Ensure unique key generation by appending the index to the name
            unique_key = f"{name}_{i}"
            
            # Exception handling for duplicate keys
            try:
                action = st.selectbox(f"Mark attendance for {name}", ["entry", "exit"], key=unique_key)
                if st.button(f"Mark {name} attendance as {action}", key=f"{unique_key}_button"):
                    mark_attendance(name, action)
            except Exception as e:
                st.error(f"Live streaming exception occurred while marking attendance for {name}: {e}")
    
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

st.markdown(""" 
    <style>
        /* Global Settings */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7fc; /* Light background for a clean look */
            color: #333333; /* Dark text color for readability */
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
            color: #2e3a87; /* Accent color for headings */
            font-weight: bold;
            margin-bottom: 10px;
        }

        /* Centering the content */
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        /* Container for sections */
        .stButton, .stCheckbox, .stRadio, .stSelectbox {
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Add rounded corners and shadows to cards */
        .stCard {
            background-color: #fff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            width: 80%;
            max-width: 1200px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .stCard:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        }

        /* Button styling with hover effects */
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
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stButton>button:hover {
            background-color: #4c6bdb;
            transform: scale(1.05);
        }

        /* Styling for input fields */
        .stTextInput, .stSelectbox, .stNumberInput {
            margin-top: 15px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
            width: 100%;
            max-width: 400px;
            transition: border-color 0.3s ease;
        }

        .stTextInput:focus, .stSelectbox:focus, .stNumberInput:focus {
            border-color: #2e3a87;
            outline: none;
        }

        /* Image styling */
        .stImage {
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
            margin-top: 20px;
            max-width: 100%;
            transition: transform 0.3s ease;
        }

        .stImage:hover {
            transform: scale(1.03);
        }

        /* Table (attendance log) styling */
        .stDataFrame {
            font-family: 'Courier New', Courier, monospace;
            border-radius: 10px;
            border: 2px solid #2e3a87;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            padding: 10px;
            background-color: #ffffff;
        }

        /* Alert boxes styling */
        .stAlert {
            border-left: 4px solid #4caf50;
            background-color: #f1f8e9;
            padding: 10px;
            margin-top: 15px;
        }

        /* Page Title */
        .page-title {
            font-size: 32px;
            font-weight: 600;
            color: #2e3a87;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 10px;
        }

        /* Custom Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #2e3a87;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background-color: #4c6bdb;
        }

        /* Customize checkbox/radio buttons */
        .stCheckbox, .stRadio {
            margin-top: 20px;
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
        This system uses **face recognition** to automatically mark attendance as people enter or leave a room.
        - **Mark Attendance**: Face recognition identifies individuals, and the system logs their entry or exit.
        - **Register New Faces**: Upload a photo and register new faces for recognition.
        - **View Attendance**: View the recorded attendance details.

        ### Instructions:
        1. First, upload known faces and add their names.
        2. Then, use the webcam feed to recognize faces and mark attendance.
        3. You can also view the attendance log anytime.
    """)
    return

# Main App Workflow
known_faces, known_names = load_known_faces()

# Choose which page to display
page = st.sidebar.selectbox("Select Page", ["Home", "Mark Attendance", "Register New Face", "View Attendance"])

if page == "Home":
    home_page()
elif page == "Mark Attendance":
    st.header("Mark Attendance")
    st.write("Start the webcam feed to begin recognizing faces.")
    
    if st.checkbox("Start Webcam"):
        stframe = st.empty()
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = recognize_multiple_faces(frame, rgb_frame, known_faces, known_names)
            
            stframe.image(frame, channels="BGR", use_column_width=True)
        
        video_capture.release()
elif page == "Register New Face":
    known_faces, known_names = register_face(known_faces, known_names)
elif page == "View Attendance":
    st.header("Attendance Log")
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("No attendance records found.")










