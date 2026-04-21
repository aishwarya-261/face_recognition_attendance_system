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
# We use a more aggressive script to catch 'q' and click the close button
components.html("""
<script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'q' || e.key === 'Q') {
            // Find ALL buttons and click the one that says 'Close'
            const buttons = doc.querySelectorAll('button');
            buttons.forEach(function(btn) {
                if (btn.innerText.includes('Close') || btn.innerText.includes('Q')) {
                    btn.click();
                }
            });
        }
    });
</script>
<script>
    // No-op script to keep the structure, but removing the hide logic
</script>
<style>
    /* Styling Take Photo button to match Image 1 (White bg, Red text, Red border) */
    [data-testid="stCameraInputButton"] {
        background-color: white !important;
        color: #ef4444 !important; /* Red-500 */
        border: 2px solid #ef4444 !important;
        border-radius: 5px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
        height: 50px !important;
    }
    [data-testid="stCameraInputButton"]:hover {
        background-color: #fee2e2 !important; /* Very light red */
        cursor: pointer;
    }
</style>
<script>
    // System recovery: Ensuring the Red/White style is enforced
    const observer = new MutationObserver(() => {
        const btns = document.querySelectorAll('[data-testid="stCameraInputButton"]');
        btns.forEach(btn => {
            btn.style.backgroundColor = "white";
            btn.style.color = "#ef4444";
            btn.style.border = "2px solid #ef4444";
            btn.style.fontWeight = "bold";
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", height=0)

# Initialize Session State
if 'show_enroll_cam' not in st.session_state:
    st.session_state.show_enroll_cam = False
if 'show_attendance_cam' not in st.session_state:
    st.session_state.show_attendance_cam = False
if 'last_recognized' not in st.session_state:
    st.session_state.last_recognized = ""
if 'marked_img' not in st.session_state:
    st.session_state.marked_img = None

# Title
st.markdown('<h1 class="main-title">Face Recognition Attendance System</h1>', unsafe_allow_html=True)

# 2-Column Layout: Left for Camera, Right for Form/Buttons
main_left, main_right = st.columns([1.5, 2], gap="large")

with main_left:
    st.markdown("### 📷 Camera View")
    
    if st.session_state.show_enroll_cam:
        st.info("💡 Capture Face for Enrollment (Static)")
        camera_photo = st.camera_input("Enrollment", label_visibility="collapsed")
        if camera_photo:
            img = Image.open(camera_photo)
            with st.spinner("Saving exactly 20 augmented samples..."):
                res = attendance_pipeline.enroll_from_image(st.session_state.get('id_input',''), st.session_state.get('name_input',''), img, 1)
            st.success(res)
            st.session_state.show_enroll_cam = False
            st.rerun()

    elif st.session_state.show_attendance_cam:
        # 🌟 BACK TO STABLE CAMERA
        st.info("💡 Look at the camera and click 'Take Photo' to log attendance.")
        attendance_camera = st.camera_input("Attendance", label_visibility="collapsed")
        
        if attendance_camera:
            img = Image.open(attendance_camera)
            with st.spinner("Recognizing..."):
                # Use optimized recognize_face (returns Status, NameText, AnnotatedImg)
                status, result, annotated = attendance_pipeline.recognize_face(img)
                
            if status == "Recognized":
                st.session_state.last_recognized = result
                st.session_state.marked_img = annotated
                # Log attendance
                first_person = result.split(",")[0].strip()
                attendance_pipeline.log_attendance(first_person)
                st.toast(f"Attendance marked for {result}!", icon="✅")
            else:
                st.session_state.marked_img = annotated
                st.error(f"{status}: {result}")
        
        # Display the result with BOXES and GREEN NAMES
        if st.session_state.marked_img is not None:
            st.subheader("Last Recognition Result:")
            st.image(st.session_state.marked_img, use_container_width=True)
            if st.button("Clear Result & Try Again"):
                st.session_state.marked_img = None
                st.rerun()
            
    else:
        st.markdown("<div style='height: 300px; border: 2px dashed #374151; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #4b5563;'>Camera is Closed. Click 'Automatic Attendance' to open.</div>", unsafe_allow_html=True)

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
            st.session_state.last_recognized = ""
            st.rerun()

st.divider()

# Discreet Admin Dashboard at the bottom
with st.expander("🔓 Admin Access (Host Only)"):
    admin_password = st.text_input("Enter Password", type="password")
    
    if admin_password == "aishwarya123":
        st.success("Access Granted")
        
        # New Danger Zone
        st.warning("⚠️ Danger Zone")
        if st.checkbox("Enable Wipe Button"):
            if st.button("🗑️ Clear All System Data (Permanent)", type="secondary"):
                res = attendance_pipeline.wipe_all_data()
                st.success(res)
                st.rerun()

        tab_logs, tab_imgs = st.tabs(["📊 Attendance Logs", "🖼️ Student Gallery"])
        
        with tab_logs:
            st.subheader("Private Attendance Records")
            attendance_file = "Attendance/Master_Attendance.csv"
            if os.path.exists(attendance_file):
                import pandas as pd
                df = pd.read_csv(attendance_file)
                # 🌟 NEW: Sort descending (newest at top)
                df_sorted = df.iloc[::-1].reset_index(drop=True)
                st.dataframe(df_sorted, use_container_width=True)
                
                # Download button
                csv = df_sorted.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Master Attendance CSV",
                    data=csv,
                    file_name='master_attendance.csv',
                    mime='text/csv',
                )
            else:
                st.info("No attendance records found on server.")

        with tab_imgs:
            st.subheader("Enrolled Student Images")
            img_path = "TrainingImage"
            if os.path.exists(img_path):
                # 🌟 IMPROVED: Scans All 20 images
                images = sorted([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))])
                if images:
                    st.write(f"Total Images in System: {len(images)}")
                    cols = st.columns(4)
                    for idx, img_name in enumerate(images):
                        with cols[idx % 4]:
                            img = Image.open(os.path.join(img_path, img_name))
                            st.image(img, caption=img_name, use_container_width=True)
                else:
                    st.info("No training images found yet.")
            else:
                st.info("Training folder is empty.")
    elif admin_password:
        st.error("Incorrect Password")

st.markdown("<p style='text-align:center; color:#4b5563;'>Core logic: 20-image augmentation | Private Logs maintained.</p>", unsafe_allow_html=True)
