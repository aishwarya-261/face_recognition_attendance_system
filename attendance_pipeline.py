import cv2
import os
import pandas as pd
import numpy as np
import datetime
import time
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pickle
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────
# MODEL SINGLETONS
# ─────────────────────────────────────────────
_mtcnn = None
_resnet = None
_trained_data = None

def get_models():
    global _mtcnn, _resnet
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=device)
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _mtcnn, _resnet

def get_trained_data(force_reload=False):
    global _trained_data
    if _trained_data is None or force_reload:
        try:
            with open("TrainingImageLabel/Trainer.pkl", 'rb') as f:
                _trained_data = pickle.load(f)
        except FileNotFoundError:
            return None
    return _trained_data

def ensure_folders():
    for f in ['TrainingImage', 'TrainingImageLabel', 'StudentDetails', 'Attendance']:
        if not os.path.exists(f):
            os.makedirs(f)

# ─────────────────────────────────────────────
# SHARED PREPROCESSING — USED EVERYWHERE
# ─────────────────────────────────────────────
def get_embedding(resnet, pil_face_crop):
    """
    Convert a PIL face crop → unit-normalized 512-d embedding.
    This EXACT function must be used during BOTH training and recognition.
    """
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # normalize to [-1, 1]
    ])
    tensor = trans(pil_face_crop.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(tensor).cpu().numpy()[0]
    # L2 normalize so cosine distance == euclidean distance on unit sphere
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

# ─────────────────────────────────────────────
# WIPE ALL
# ─────────────────────────────────────────────
def wipe_all_data():
    import shutil
    ensure_folders()
    for folder in ['TrainingImage', 'TrainingImageLabel', 'StudentDetails', 'Attendance']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    ensure_folders()
    global _trained_data
    _trained_data = None
    return "All data successfully cleared. You can start fresh now."

# ─────────────────────────────────────────────
# ENROLLMENT (Web — photo upload)
# ─────────────────────────────────────────────
def enroll_from_image(enrollment_id, name, image, sample_num=None):
    """Enroll a student from a single snapshot. Generates 20 augmented face crops."""
    ensure_folders()
    try:
        int(enrollment_id)
    except Exception:
        return "Error: Enrollment ID must be numeric."

    if not name.strip():
        return "Error: Name cannot be empty."

    pil_img = image.convert('RGB')
    mtcnn, _ = get_models()
    boxes, _ = mtcnn.detect(pil_img)

    if boxes is None:
        return "Error: No face detected. Please look directly at the camera and try again."

    # Pick the largest / most prominent face
    best_idx = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]))
    x1, y1, x2, y2 = [int(b) for b in boxes[best_idx]]
    w, h = x2 - x1, y2 - y1
    pad = 0.15
    x1 = max(0, x1 - int(w * pad))
    y1 = max(0, y1 - int(h * pad))
    x2 = min(pil_img.width,  x2 + int(w * pad))
    y2 = min(pil_img.height, y2 + int(h * pad))
    face_crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160), Image.LANCZOS)

    # ── Deterministic 20-sample augmentation ──────────────────────────
    # Each group of 4 applies a guaranteed distinct effect so the gallery
    # clearly shows: Original, Flip, Bright, Dark, Zoom-In, Zoom-Out
    # (5 rounds × 4 variants = 20 images)
    def make_augmented_samples(base_img, total=20):
        import random
        from PIL import ImageEnhance
        samples = []

        for i in range(total):
            img = base_img.copy()
            slot = i % 5  # 0=original, 1=flip, 2=brighten, 3=darken, 4=zoom

            if slot == 0:
                # Original with tiny random translate
                tx = random.randint(-5, 5)
                ty = random.randint(-5, 5)
                tmp = transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))(img)
                img = tmp

            elif slot == 1:
                # Horizontal flip (always applies)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            elif slot == 2:
                # Brighten (+40-60%)
                factor = random.uniform(1.4, 1.6)
                img = ImageEnhance.Brightness(img).enhance(factor)
                img = ImageEnhance.Contrast(img).enhance(random.uniform(1.1, 1.3))

            elif slot == 3:
                # Darken (50-70% brightness)
                factor = random.uniform(0.5, 0.7)
                img = ImageEnhance.Brightness(img).enhance(factor)
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

            elif slot == 4:
                # Zoom in (scale 1.15–1.25) — crop center
                scale = random.uniform(1.15, 1.25)
                new_w = int(160 * scale)
                new_h = int(160 * scale)
                img_big = img.resize((new_w, new_h), Image.LANCZOS)
                left = (new_w - 160) // 2
                top = (new_h - 160) // 2
                img = img_big.crop((left, top, left + 160, top + 160))

            # Always resize to final 160×160
            img = img.resize((160, 160), Image.LANCZOS)
            samples.append(img)

        return samples

    aug_samples = make_augmented_samples(face_crop, total=20)

    saved = 0
    for i, out in enumerate(aug_samples, start=1):
        try:
            fname = f"TrainingImage/{name}.{enrollment_id}.{i}.jpg"
            out.save(fname, "JPEG", quality=95)
            saved += 1
        except Exception as e:
            print(f"Save error sample {i}: {e}")

    # Update StudentDetails.csv
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        pd.DataFrame([[enrollment_id, name]], columns=['Id', 'Name']).to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)
        df['Id'] = df['Id'].astype(str)
        if str(enrollment_id) not in df['Id'].values:
            pd.DataFrame([[enrollment_id, name]], columns=['Id', 'Name']).to_csv(
                df_path, mode='a', header=False, index=False)

    if saved >= 20:
        return f"✅ Enrolled {name} successfully! {saved} face samples saved."
    else:
        return f"⚠️ Warning: Only saved {saved}/20 samples."

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_images():
    ensure_folders()
    path = "TrainingImage"
    image_files = [f for f in os.listdir(path) if f.lower().endswith('.jpg')]

    if not image_files:
        return "No images found. Please enroll students first."

    _, resnet = get_models()

    embeddings = []
    ids = []
    skipped = 0

    for fname in image_files:
        try:
            # Filename format: Name.ID.SampleNum.jpg
            parts = fname.split('.')
            student_id = int(parts[1])
            img = Image.open(os.path.join(path, fname))
            emb = get_embedding(resnet, img)
            embeddings.append(emb)
            ids.append(student_id)
        except Exception as e:
            print(f"Skipped {fname}: {e}")
            skipped += 1

    if not embeddings:
        return "Error: Could not process any images. Check image file names."

    embeddings = np.array(embeddings)
    ids = np.array(ids)

    # Build per-student PROTOTYPE (mean embedding) for fast, accurate inference
    unique_ids = np.unique(ids)
    prototypes = {}
    for uid in unique_ids:
        mask = ids == uid
        mean_emb = np.mean(embeddings[mask], axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)  # re-normalize
        prototypes[uid] = mean_emb

    model_data = {
        "embeddings": embeddings,
        "ids": ids,
        "prototypes": prototypes,   # ← NEW: used for recognition
    }

    global _trained_data
    _trained_data = None  # Force cache reload next time

    with open("TrainingImageLabel/Trainer.pkl", 'wb') as f:
        pickle.dump(model_data, f)

    return (f"✅ Model trained on {len(embeddings)} images ({len(unique_ids)} students). "
            f"Ready for attendance! ({skipped} images skipped)")

# ─────────────────────────────────────────────
# RECOGNITION (Web — snapshot)
# ─────────────────────────────────────────────
def recognize_face(image):
    """Recognize face(s) from a snapshot image. Returns (status, label, annotated_image)."""
    ensure_folders()
    model_data = get_trained_data()
    if model_data is None:
        return "Error", "Model not found. Please run Train Model first.", image

    prototypes = model_data.get("prototypes")
    if prototypes is None:
        # Fallback: build prototypes from raw embeddings (old pkl format)
        saved_embeddings = model_data["embeddings"]
        saved_ids = model_data["ids"]
        prototypes = {}
        for uid in np.unique(saved_ids):
            mask = saved_ids == uid
            mean_emb = np.mean(saved_embeddings[mask], axis=0)
            prototypes[uid] = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)

    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        return "Error", "Student details not found.", image
    df = pd.read_csv(df_path)
    df['Id'] = df['Id'].astype(str)

    mtcnn, resnet = get_models()
    img_rgb = image.convert('RGB')
    draw = ImageDraw.Draw(img_rgb)
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is None:
        return "No Face", "No face detected in the image.", img_rgb

    results_list = []

    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

        try:
            # Pad crop same way as enrollment (15%)
            w, h = x2 - x1, y2 - y1
            pad = 0.15
            cx1 = max(0, x1 - int(w * pad))
            cy1 = max(0, y1 - int(h * pad))
            cx2 = min(img_rgb.width,  x2 + int(w * pad))
            cy2 = min(img_rgb.height, y2 + int(h * pad))
            face_crop = img_rgb.crop((cx1, cy1, cx2, cy2))

            emb = get_embedding(resnet, face_crop)

            # Compare against each student's PROTOTYPE embedding
            best_id = None
            best_dist = float('inf')
            for uid, proto in prototypes.items():
                dist = np.linalg.norm(emb - proto)
                if dist < best_dist:
                    best_dist = dist
                    best_id = uid

            # Threshold: 0.9 works well for L2-norm unit embeddings
            # Lower = stricter. 0.9 is generous enough for real-world lighting variance.
            THRESHOLD = 0.9
            if best_dist < THRESHOLD:
                name_row = df.loc[df['Id'] == str(best_id)]['Name'].values
                name = name_row[0] if len(name_row) > 0 else "Unknown"
                label = f"{best_id}-{name}"
                draw.text((x1, max(0, y1 - 35)), label, fill="lime")
                results_list.append(label)
            else:
                draw.text((x1, max(0, y1 - 35)), "Unknown", fill="red")

        except Exception as e:
            print(f"Recognition error: {e}")
            continue

    final_status = "Recognized" if results_list else "No Match"
    return final_status, ", ".join(results_list), img_rgb

# ─────────────────────────────────────────────
# ATTENDANCE LOGGING
# ─────────────────────────────────────────────
def log_attendance(display_text):
    """Log attendance for a recognized student (format: 'ID-Name')."""
    try:
        parts = display_text.split("-", 1)
        if len(parts) < 2:
            return False
        student_id = parts[0].strip()
        name = parts[1].strip()

        ist_now = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        date = ist_now.strftime('%Y-%m-%d')
        timestamp = ist_now.strftime('%H:%M:%S')

        ensure_folders()
        fname = "Attendance/Master_Attendance.csv"
        row = pd.DataFrame([[student_id, name, date, timestamp]],
                           columns=['Id', 'Name', 'Date', 'Time'])
        if os.path.exists(fname):
            row.to_csv(fname, mode='a', header=False, index=False)
        else:
            row.to_csv(fname, mode='w', header=True, index=False)
        return True
    except Exception as e:
        print(f"Logging error: {e}")
        return False


# ─────────────────────────────────────────────
# LEGACY desktop mode (kept for reference)
# ─────────────────────────────────────────────
def capture_images(enrollment_id, name):
    ensure_folders()
    try:
        int(enrollment_id)
    except ValueError:
        return "Error: Enrollment ID must be numeric."

    mtcnn_single = MTCNN(keep_all=False, device=device)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "Error: Could not open camera."

    sampleNum = 0
    wname = 'Face Capturing (press Q to stop)'
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        boxes, _ = mtcnn_single.detect(pil_img)
        if boxes is not None:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    sampleNum += 1
                    cv2.imwrite(f"TrainingImage/{name}.{enrollment_id}.{sampleNum}.jpg", face_img)
        cv2.imshow(wname, frame)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 20:
            break
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1:
            break
    cam.release()
    cv2.destroyAllWindows()
    return f"Captured {sampleNum} images for {name}."


def automatic_attendance():
    ensure_folders()
    model_data = get_trained_data()
    if model_data is None:
        return "Model not found. Please train images first."

    prototypes = model_data.get("prototypes")
    if prototypes is None:
        saved_embeddings = model_data["embeddings"]
        saved_ids = model_data["ids"]
        prototypes = {}
        for uid in np.unique(saved_ids):
            mask = saved_ids == uid
            mean_emb = np.mean(saved_embeddings[mask], axis=0)
            prototypes[uid] = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)

    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        return "Student details not found."
    df = pd.read_csv(df_path)
    df['Id'] = df['Id'].astype(str)

    _, resnet = get_models()
    mtcnn_live = MTCNN(keep_all=True, device=device)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        return "Error: Could not open camera."

    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])
    THRESHOLD = 0.9

    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    first_frame = True
    wname = 'Taking Attendance (Press Q to finish)'

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        boxes, _ = mtcnn_live.detect(pil_img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                h_f, w_f = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                try:
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    tensor = trans(face_crop.convert('RGB')).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = resnet(tensor).cpu().numpy()[0]
                    emb = emb / (np.linalg.norm(emb) + 1e-10)

                    best_id, best_dist = None, float('inf')
                    for uid, proto in prototypes.items():
                        d = np.linalg.norm(emb - proto)
                        if d < best_dist:
                            best_dist, best_id = d, uid

                    if best_dist < THRESHOLD:
                        name_row = df.loc[df['Id'] == str(best_id)]['Name'].values
                        name = name_row[0] if len(name_row) > 0 else "Unknown"
                        label = f"{best_id}-{name}"
                        color = (0, 255, 0)
                        ist = datetime.datetime.now(
                            datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
                        attendance.loc[len(attendance)] = [
                            str(best_id), name,
                            ist.strftime('%Y-%m-%d'), ist.strftime('%H:%M:%S')]
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)), font, 0.8, color, 2)
                except Exception as e:
                    print(e)

        cv2.imshow(wname, frame)
        if first_frame:
            cv2.waitKey(1)
            first_frame = False
        key = cv2.waitKey(100) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
    attendance = attendance[attendance['Id'] != 'Unknown']

    if len(attendance) > 0:
        master = "Attendance/Master_Attendance.csv"
        if os.path.exists(master):
            attendance.to_csv(master, mode='a', header=False, index=False)
        else:
            attendance.to_csv(master, mode='w', header=True, index=False)
        return f"Attendance marked for {len(attendance)} student(s)."
    else:
        return "No recognized faces to mark attendance."
