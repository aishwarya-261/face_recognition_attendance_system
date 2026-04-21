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
page = st.sidebar.radio("Go to", ["1. Enroll Student", "2. Train Model", "3. Mark Attendance", "4. View Data & Logs"])

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

elif page == "4. View Data & Logs":
    st.header("Step 4: View Enrolled Students and Attendance Logs")
    
    tab1, tab2 = st.tabs(["📊 Attendance Logs", "🖼️ Student Gallery"])
    
    with tab1:
        st.subheader("Attendance Records")
        attendance_file = "Attendance/Master_Attendance.csv"
        if os.path.exists(attendance_file):
            import pandas as pd
            df = pd.read_csv(attendance_file)
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Attendance CSV",
                data=csv,
                file_name='attendance_report.csv',
                mime='text/csv',
            )
        else:
            st.info("No attendance records found yet.")

    with tab2:
        st.subheader("Captured Training Images")
        img_path = "TrainingImage"
        if os.path.exists(img_path):
            images = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))]
            if images:
                cols = st.columns(4)
                for idx, img_name in enumerate(images):
                    with cols[idx % 4]:
                        img = Image.open(os.path.join(img_path, img_name))
                        st.image(img, caption=img_name, use_container_width=True)
            else:
                st.info("No training images found yet.")
        else:
            st.info("Training folder not found.")

st.divider()
st.caption("Built with PyTorch, FaceNet, and Streamlit. Designed for stable cloud deployment.")
