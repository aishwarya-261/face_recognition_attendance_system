import streamlit as st
import attendance_pipeline
from PIL import Image
import os

# Page Config - Hide sidebar and set theme
st.set_page_config(page_title="Face Recognition Attendance", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to match the original Tkinter UI
st.markdown("""
    <style>
    /* Dark Background */
    .stApp {
        background-color: #111827;
        color: white;
    }
    
    /* Main Title */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 40px;
        margin-top: 20px;
        margin-bottom: 40px;
    }
    
    /* Enrollment Frame (Card) */
    .enroll-frame {
        background-color: #1f2937;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        max-width: 600px;
        margin: auto;
        border: 1px solid #374151;
    }
    
    /* Buttons Row */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 50px;
    }
    
    /* Specific Button Styling */
    div.stButton > button {
        height: 60px;
        font-weight: 700;
        border-radius: 5px;
        border: none;
        color: white;
        font-size: 18px;
        width: 100%;
    }
    
    /* Button 1 & 2 Colors (Blue) */
    .st-emotion-cache-unio6o.e1nzvjt11 { /* Take Images & Train Model */
        background-color: #2563eb !important;
    }
    
    /* Button 3 Color (Green) */
    .st-emotion-cache-163o64k.e1nzvjt11 { /* Attendance */
        background-color: #10b981 !important;
    }

    /* Input labels color */
    .stMarkdown p {
        color: #d1d5db;
        font-weight: 600;
    }
    
    /* Hide specific Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">Face Recognition Attendance System</h1>', unsafe_allow_html=True)

# Main Dashboard Container
with st.container():
    # Enrollment "Frame" (Like the card in screenshot)
    st.markdown('<div class="enroll-frame">', unsafe_allow_html=True)
    
    col_label, col_input = st.columns([1, 2])
    with col_label:
        st.markdown("<br><p style='margin-top:10px;'>Enrollment ID:</p>", unsafe_allow_html=True)
        st.markdown("<br><p style='margin-top:15px;'>Student Name:</p>", unsafe_allow_html=True)
        
    with col_input:
        enroll_id = st.text_input("", label_visibility="collapsed", key="id_input", placeholder="Enter ID")
        student_name = st.text_input("", label_visibility="collapsed", key="name_input", placeholder="Enter Name")
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Status/Output Message Area
    status_container = st.empty()

    # Buttons Row (Mimicking the screenshot layout)
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        # Step 1: Capture
        camera_photo = st.camera_input("1. Take Images", label_visibility="collapsed")
        if st.button("1. Take Images", use_container_width=True, type="primary"):
            if not enroll_id or not student_name:
                status_container.error("Please enter both ID and Name first.")
            elif camera_photo is None:
                status_container.warning("Please capture a photo using the camera above.")
            else:
                image = Image.open(camera_photo)
                with st.spinner("Processing 20 augmented samples..."):
                    res = attendance_pipeline.enroll_from_image(enroll_id, student_name, image, 1)
                status_container.success(res)

    with btn_col2:
        # Step 2: Train
        if st.button("2. Train Model", use_container_width=True, type="primary"):
            with st.spinner("Training..."):
                res = attendance_pipeline.train_images()
            status_container.success(res)

    with btn_col3:
        # Step 3: Attendance
        attendance_camera = st.camera_input("3. Automatic Attendance", label_visibility="collapsed")
        if st.button("3. Automatic Attendance", use_container_width=True):
            if attendance_camera is None:
                status_container.warning("Capture face for attendance.")
            else:
                image = Image.open(attendance_camera)
                with st.spinner("Recognizing..."):
                    status, result = attendance_pipeline.recognize_face(image)
                if status == "Recognized":
                    status_container.success(f"Marked: {result}")
                else:
                    status_container.error(f"{status}: {result}")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4b5563;'>Original logic maintained (20-image augmentation).</p>", unsafe_allow_html=True)
