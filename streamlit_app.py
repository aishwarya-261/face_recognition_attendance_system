import streamlit as st
import attendance_pipeline
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="Face Recognition Attendance", page_icon="🎓", layout="wide")

st.title("🎓 Face Recognition Attendance System")
st.markdown("Register students, train the AI model, and mark attendance securely.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Enroll Student", "2. Train Model", "3. Mark Attendance"])

if page == "1. Enroll Student":
    st.header("Step 1: Enroll a New Student")
    col1, col2 = st.columns(2)
    
    with col1:
        enroll_id = st.text_input("Enrollment ID", placeholder="e.g. 101")
        student_name = st.text_input("Student Name", placeholder="e.g. John Doe")
        camera_photo = st.camera_input("Capture Face for Enrollment")
        
    with col2:
        if st.button("Register Student", type="primary"):
            if not enroll_id or not student_name:
                st.error("Please enter both ID and Name")
            elif camera_photo is None:
                st.error("Please capture a photo first")
            else:
                image = Image.open(camera_photo)
                with st.spinner("Processing image..."):
                    res = attendance_pipeline.enroll_from_image(enroll_id, student_name, image, 1)
                if "Success" in res:
                    st.success(res)
                else:
                    st.error(res)

elif page == "2. Train Model":
    st.header("Step 2: Train the AI Model")
    st.info("Click the button below to train the model on all enrolled students. This might take a minute.")
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... this may take a moment."):
            res = attendance_pipeline.train_images()
        if "Successfully" in res:
            st.success(res)
        else:
            st.error(res)

elif page == "3. Mark Attendance":
    st.header("Step 3: Mark Automatic Attendance")
    col1, col2 = st.columns(2)
    
    with col1:
        camera_photo = st.camera_input("Capture Face for Attendance")
        
    with col2:
        if st.button("✅ Mark Attendance", type="primary"):
            if camera_photo is None:
                st.error("Please capture a photo first")
            else:
                image = Image.open(camera_photo)
                with st.spinner("Recognizing..."):
                    status, result = attendance_pipeline.recognize_face(image)
                
                if status == "Recognized":
                    st.success(f"Successfully marked: {result}")
                elif status == "No Face":
                    st.warning("No face detected in the photo.")
                else:
                    st.error(f"Recognition Status: {status} - {result}")

st.divider()
st.caption("Built with PyTorch, FaceNet, and Streamlit. Designed for stable cloud deployment.")
