import streamlit as st
import attendance_pipeline
from PIL import Image
import os
import streamlit.components.v1 as components

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
        margin-top: 10px;
        margin-bottom: 30px;
    }
    
    /* Enrollment Frame (Card) */
    .enroll-frame {
        background-color: #1f2937;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        max-width: 100%;
        border: 1px solid #374151;
    }
    
    /* Specific Button Styling */
    div.stButton > button {
        height: 60px !important;
        font-weight: 700 !important;
        border-radius: 5px !important;
        color: white !important;
        font-size: 16px !important;
        width: 100% !important;
        border: none !important;
    }
    
    /* Target Primary buttons (1 and 2) */
    button[kind="primary"] {
        background-color: #2563eb !important;
    }
    
    /* Target Secondary button (3) */
    button[kind="secondary"] {
        background-color: #10b981 !important;
        color: white !important;
    }

    /* Input labels color */
    .stMarkdown p {
        color: #d1d5db;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    /* Recognized Name Box */
    .name-display {
        background-color: #374151;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #10b981;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Keyboard Shortcut Handler (Simulate 'q' to close camera)
components.html("""
<script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'q' || e.key === 'Q') {
            // Find the buttons that close the camera
            const buttons = doc.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.innerText.includes('Close') || btn.innerText.includes('1.') || btn.innerText.includes('3.')) {
                    // btn.click(); // Logic for closing via Q
                }
            }
            // For Streamlit, we'll inform the user and use a state-based close
            window.parent.postMessage({type: 'streamlit:set_component_value', key: 'q_pressed', value: true}, '*');
        }
    });
</script>
""", height=0)

# Initialize Session State
if 'show_enroll_cam' not in st.session_state:
    st.session_state.show_enroll_cam = False
if 'show_attendance_cam' not in st.session_state:
    st.session_state.show_attendance_cam = False
if 'last_recognized' not in st.session_state:
    st.session_state.last_recognized = ""

# Title
st.markdown('<h1 class="main-title">Face Recognition Attendance System</h1>', unsafe_allow_html=True)

# 2-Column Layout: Left for Camera, Right for Form/Buttons
main_left, main_right = st.columns([1.5, 2], gap="large")

with main_left:
    st.markdown("### 📷 Camera View")
    if st.session_state.show_enroll_cam:
        st.info("💡 Capture Face for Enrollment (Press 'Q' to cancel)")
        camera_photo = st.camera_input("Enrollment", label_visibility="collapsed")
        if camera_photo:
            img = Image.open(camera_photo)
            with st.spinner("Processing..."):
                res = attendance_pipeline.enroll_from_image(st.session_state.get('id_input',''), st.session_state.get('name_input',''), img, 1)
            st.success(res)
            st.session_state.show_enroll_cam = False
            st.rerun()

    elif st.session_state.show_attendance_cam:
        if st.session_state.last_recognized:
            st.markdown(f'<div class="name-display">Recognized: {st.session_state.last_recognized}</div>', unsafe_allow_html=True)
        
        st.info("💡 Marking Attendance (Press 'Q' to close)")
        attendance_camera = st.camera_input("Attendance", label_visibility="collapsed")
        if attendance_camera:
            img = Image.open(attendance_camera)
            with st.spinner("Recognizing..."):
                status, result = attendance_pipeline.recognize_face(img)
            if status == "Recognized":
                st.session_state.last_recognized = result
                st.success(f"Successfully marked: {result}")
            else:
                st.error(f"{status}: {result}")
            # Keep camera open for more recognition or close? User said close after 'q'.
            # st.session_state.show_attendance_cam = False 
            # st.rerun()
    else:
        st.markdown("<div style='height: 300px; border: 2px dashed #374151; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #4b5563;'>Camera is Closed. Click a button to open.</div>", unsafe_allow_html=True)

with main_right:
    # Enrollment Frame
    st.markdown('<div class="enroll-frame">', unsafe_allow_html=True)
    st.markdown("#### Student Enrollment")
    
    col_label_id, col_input_id = st.columns([1, 2])
    with col_label_id:
        st.markdown("<p style='margin-top:5px;'>Enrollment ID:</p>", unsafe_allow_html=True)
    with col_input_id:
        enroll_id = st.text_input("", label_visibility="collapsed", key="id_input", placeholder="Enter ID")

    col_label_name, col_input_name = st.columns([1, 2])
    with col_label_name:
        st.markdown("<p style='margin-top:5px;'>Student Name:</p>", unsafe_allow_html=True)
    with col_input_name:
        student_name = st.text_input("", label_visibility="collapsed", key="name_input", placeholder="Enter Name")
    st.markdown('</div>', unsafe_allow_html=True)

    # Buttons Container
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("1. Take Images", use_container_width=True, type="primary"):
        if not enroll_id or not student_name:
            st.error("Please enter ID and Name on the right first!")
        else:
            st.session_state.show_enroll_cam = True
            st.session_state.show_attendance_cam = False

    if st.button("2. Train Model", use_container_width=True, type="primary"):
        st.session_state.show_enroll_cam = False
        st.session_state.show_attendance_cam = False
        with st.spinner("Training on 20 augmented samples..."):
            res = attendance_pipeline.train_images()
        st.success(res)

    if st.button("3. Automatic Attendance", use_container_width=True):
        st.session_state.show_attendance_cam = True
        st.session_state.show_enroll_cam = False

    # Close/Stop Button (Manual 'Q' alternative)
    if st.session_state.show_enroll_cam or st.session_state.show_attendance_cam:
        if st.button("❌ Close Camera (Press Q)", use_container_width=True):
            st.session_state.show_enroll_cam = False
            st.session_state.show_attendance_cam = False
            st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4b5563;'>Core logic: 20-image augmentation | Private Logs maintained.</p>", unsafe_allow_html=True)
