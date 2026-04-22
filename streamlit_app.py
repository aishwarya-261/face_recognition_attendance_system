import streamlit as st
import attendance_pipeline
from PIL import Image
import os
import pandas as pd
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* ── Base ── */
    .stApp { background-color: #111827; color: #f9fafb; }

    /* ── Force ALL text white so nothing is invisible ── */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp h1, .stApp h2, .stApp h3,
    .stApp li, .stApp a {
        color: #f9fafb !important;
    }

    /* ── Inputs: text fields ── */
    .stTextInput input {
        background-color: #374151 !important;
        color: #f9fafb !important;
        border: 1px solid #6b7280 !important;
        border-radius: 6px !important;
    }
    .stTextInput input::placeholder { color: #9ca3af !important; }
    .stTextInput label { color: #f9fafb !important; font-weight: 600 !important; }

    /* ── Password input ── */
    .stTextInput input[type="password"] {
        background-color: #374151 !important;
        color: #f9fafb !important;
    }

    /* ── Selectbox / dropdown ── */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #374151 !important;
        color: #f9fafb !important;
        border-color: #6b7280 !important;
    }
    .stSelectbox span { color: #f9fafb !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        color: #d1d5db !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #10b981 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        color: #f9fafb !important;
        font-weight: 700 !important;
        background-color: #1f2937 !important;
    }
    .streamlit-expanderContent {
        background-color: #1f2937 !important;
    }

    /* ── Checkbox ── */
    .stCheckbox label { color: #f9fafb !important; }
    .stCheckbox span { color: #f9fafb !important; }

    /* ── Dataframe / table ── */
    .stDataFrame { color: #f9fafb !important; }

    /* ── Info / warning / error / success boxes ── */
    .stAlert p { color: #111827 !important; }
    div[data-testid="stNotificationContentError"] p,
    div[data-testid="stNotificationContentWarning"] p,
    div[data-testid="stNotificationContentInfo"] p,
    div[data-testid="stNotificationContentSuccess"] p { color: #111827 !important; }

    /* ── Status boxes (custom) ── */
    .status-box {
        background-color: #1f2937;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        font-size: 14px;
        color: #f9fafb !important;
        margin-bottom: 12px;
    }

    /* ── Enrollment card ── */
    .enroll-frame {
        background-color: #1f2937;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #374151;
    }

    /* ── Main title ── */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 40px;
        color: #f9fafb !important;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* ── Sidebar / main buttons ── */
    div.stButton > button {
        height: 55px !important;
        font-weight: 700 !important;
        border-radius: 5px !important;
        color: white !important;
        font-size: 15px !important;
        width: 100% !important;
        border: none !important;
    }

    /* ── Secondary buttons (Close Camera, Wipe etc.) ── */
    div.stButton > button[kind="secondary"] {
        background-color: #7f1d1d !important;
        color: white !important;
        border: 1px solid #ef4444 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #991b1b !important;
    }

    /* ── Success / Warning / Error / Info alerts ── */
    .stAlert, .stAlert > div, .stAlert p,
    [data-testid="stNotification"] p,
    [data-testid="stNotification"] span,
    [data-testid="stNotificationContentSuccess"] p,
    [data-testid="stNotificationContentSuccess"] span,
    [data-testid="stNotificationContentWarning"] p,
    [data-testid="stNotificationContentWarning"] span,
    [data-testid="stNotificationContentError"] p,
    [data-testid="stNotificationContentError"] span,
    [data-testid="stNotificationContentInfo"] p,
    [data-testid="stNotificationContentInfo"] span {
        color: #111827 !important;
    }

    /* ── Camera button: always Red/White ── */
    div[data-testid="stCameraInput"] button {
        background-color: white !important;
        color: #ff4b4b !important;
        border: 2px solid #ff4b4b !important;
    }
    div[data-testid="stCameraInput"] button span {
        color: #ff4b4b !important;
        visibility: visible !important;
    }

    /* ── Toast ── */
    [data-testid="stToast"] { color: #111827 !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# JS: 'Q' key → Take Photo then Close
# ─────────────────────────────────────────────────────────
components.html("""
<script>
  const doc = window.parent.document;
  doc.addEventListener('keydown', function(e) {
    if (e.key !== 'q' && e.key !== 'Q') return;
    const buttons = doc.querySelectorAll('button');
    let captureBtn = null, closeBtn = null;
    for (let btn of buttons) {
      if (btn.innerText.includes('Take Photo')) captureBtn = btn;
      if (btn.innerText.includes('Close Camera') || btn.innerText.includes('❌')) closeBtn = btn;
    }
    if (captureBtn) {
      captureBtn.click();
      setTimeout(() => { if (closeBtn) closeBtn.click(); }, 300);
    } else if (closeBtn) {
      closeBtn.click();
    }
  });

  function fixCameraBtn() {
    doc.querySelectorAll('div[data-testid="stCameraInput"] button').forEach(btn => {
      btn.style.backgroundColor = 'white';
      btn.style.color = '#ff4b4b';
      btn.style.borderColor = '#ff4b4b';
      btn.querySelectorAll('span').forEach(s => {
        s.style.color = '#ff4b4b';
        s.style.visibility = 'visible';
      });
    });
  }
  setInterval(fixCameraBtn, 600);
</script>
""", height=0)

# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────
for key, default in {
    'show_enroll_cam': False,
    'show_attendance_cam': False,
    'last_recognized': '',
    'marked_img': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────
# SYSTEM STATUS BAR  ← NEW: shows user exactly what state system is in
# ─────────────────────────────────────────────────────────
def get_system_status():
    attendance_pipeline.ensure_folders()
    img_count = len([f for f in os.listdir("TrainingImage") if f.endswith('.jpg')]) \
        if os.path.exists("TrainingImage") else 0
    model_exists = os.path.exists("TrainingImageLabel/Trainer.pkl")
    student_count = 0
    if os.path.exists("StudentDetails/StudentDetails.csv"):
        try:
            df = pd.read_csv("StudentDetails/StudentDetails.csv")
            student_count = len(df)
        except Exception:
            student_count = 0
    return img_count, model_exists, student_count

img_count, model_exists, student_count = get_system_status()

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    icon = "✅" if student_count > 0 else "❌"
    st.markdown(f"<div class='status-box'>{icon} <b>Students Enrolled:</b> {student_count}</div>", unsafe_allow_html=True)
with col_s2:
    icon = "✅" if img_count > 0 else "❌"
    st.markdown(f"<div class='status-box'>{icon} <b>Training Images:</b> {img_count}</div>", unsafe_allow_html=True)
with col_s3:
    icon = "✅" if model_exists else "❌"
    label = "Model Ready" if model_exists else "Model NOT trained — click Train Model!"
    st.markdown(f"<div class='status-box'>{icon} <b>Model:</b> {label}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">Face Recognition Attendance System</h1>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────
main_left, main_right = st.columns([1.5, 2], gap="large")

with main_left:
    st.markdown("### 📷 Camera View")

    if st.session_state.show_enroll_cam:
        st.info("💡 Look at the camera and click 'Take Photo' (or press Q)")
        camera_photo = st.camera_input("Enrollment", label_visibility="collapsed")
        if camera_photo:
            img = Image.open(camera_photo)
            eid = st.session_state.get('id_input', '')
            ename = st.session_state.get('name_input', '')
            if not eid or not ename:
                st.error("Please enter ID and Name first!")
            else:
                with st.spinner("Detecting face & saving 20 augmented samples..."):
                    res = attendance_pipeline.enroll_from_image(eid, ename, img, 1)
                if res.startswith("✅"):
                    st.success(res)
                    st.session_state.show_enroll_cam = False
                    st.rerun()
                else:
                    st.error(res)

    elif st.session_state.show_attendance_cam:
        if not model_exists:
            st.error("⚠️ No trained model found! Please enroll students and click 'Train Model' first.")
        else:
            st.info("💡 Look at the camera and click 'Take Photo' (or press Q)")
            attendance_camera = st.camera_input("Attendance", label_visibility="collapsed")

            if attendance_camera:
                img = Image.open(attendance_camera)
                with st.spinner("Recognizing face..."):
                    status, result, annotated = attendance_pipeline.recognize_face(img)

                if status == "Recognized":
                    st.session_state.last_recognized = result
                    st.session_state.marked_img = annotated
                    first_person = result.split(",")[0].strip()
                    attendance_pipeline.log_attendance(first_person)
                    st.toast(f"✅ Attendance marked for: {result}", icon="✅")
                elif status == "No Face":
                    st.session_state.marked_img = annotated
                    st.warning("No face detected. Please look directly at the camera.")
                else:
                    st.session_state.marked_img = annotated
                    st.warning(f"Face detected but not recognized. Status: {status} — {result}")

            if st.session_state.marked_img is not None:
                st.subheader("Last Recognition Result:")
                st.image(st.session_state.marked_img, use_container_width=True)
                if st.session_state.last_recognized:
                    st.success(f"✅ Recognized: **{st.session_state.last_recognized}**")
                if st.button("Clear & Try Again"):
                    st.session_state.marked_img = None
                    st.session_state.last_recognized = ''
                    st.rerun()
    else:
        st.markdown(
            "<div style='height:300px;border:2px dashed #374151;border-radius:10px;"
            "display:flex;align-items:center;justify-content:center;color:#4b5563;'>"
            "Camera is Closed. Use buttons on the right to begin.</div>",
            unsafe_allow_html=True
        )

with main_right:
    st.markdown('<div class="enroll-frame">', unsafe_allow_html=True)
    st.markdown("#### Student Enrollment")

    col_label_id, col_input_id = st.columns([1, 2])
    with col_label_id:
        st.markdown("<p style='margin-top:5px;'>Enrollment ID:</p>", unsafe_allow_html=True)
    with col_input_id:
        enroll_id = st.text_input("", label_visibility="collapsed", key="id_input", placeholder="e.g. 101")

    col_label_name, col_input_name = st.columns([1, 2])
    with col_label_name:
        st.markdown("<p style='margin-top:5px;'>Student Name:</p>", unsafe_allow_html=True)
    with col_input_name:
        student_name = st.text_input("", label_visibility="collapsed", key="name_input", placeholder="e.g. Aishwarya")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("1. Take Images", use_container_width=True, type="primary"):
        if not enroll_id or not student_name:
            st.error("Please enter both ID and Name before taking images!")
        else:
            st.session_state.show_enroll_cam = True
            st.session_state.show_attendance_cam = False

    if st.button("2. Train Model", use_container_width=True, type="primary"):
        st.session_state.show_enroll_cam = False
        st.session_state.show_attendance_cam = False
        if img_count == 0:
            st.error("No images to train on! Please enroll students first.")
        else:
            with st.spinner(f"Training model on {img_count} images..."):
                res = attendance_pipeline.train_images()
            st.success(res)
            st.rerun()   # Refresh status bar

    if st.button("3. Automatic Attendance", use_container_width=True, type="primary"):
        if not model_exists:
            st.error("⚠️ Please train the model first (Step 2)!")
        else:
            st.session_state.show_attendance_cam = True
            st.session_state.show_enroll_cam = False

    if st.session_state.show_enroll_cam or st.session_state.show_attendance_cam:
        if st.button("❌ Close Camera (Press Q)", use_container_width=True):
            st.session_state.show_enroll_cam = False
            st.session_state.show_attendance_cam = False
            st.session_state.last_recognized = ""
            st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────
# ADMIN DASHBOARD
# ─────────────────────────────────────────────────────────
with st.expander("🔓 Admin Access (Host Only)"):
    admin_password = st.text_input("Enter Password", type="password")

    if admin_password == "aishwarya123":
        st.success("Access Granted ✅")

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
                df = pd.read_csv(attendance_file)
                df_sorted = df.iloc[::-1].reset_index(drop=True)
                st.dataframe(df_sorted, use_container_width=True)
                csv = df_sorted.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Master Attendance CSV",
                    data=csv,
                    file_name='master_attendance.csv',
                    mime='text/csv',
                )
            else:
                st.info("No attendance records found.")

        with tab_imgs:
            st.subheader("Enrolled Student Images")
            img_path = "TrainingImage"
            if os.path.exists(img_path):
                images = sorted([f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png'))])
                if images:
                    st.write(f"Total Images: {len(images)}")
                    cols = st.columns(8)
                    for idx, img_name in enumerate(images):
                        with cols[idx % 8]:
                            img = Image.open(os.path.join(img_path, img_name))
                            st.image(img, caption=img_name[:12], use_container_width=True)
                else:
                    st.info("No training images found.")
            else:
                st.info("Training folder is empty.")
    elif admin_password:
        st.error("Incorrect Password")

st.markdown(
    "<p style='text-align:center;color:#4b5563;'>20-image augmentation | IST timestamps | Private logs</p>",
    unsafe_allow_html=True
)
