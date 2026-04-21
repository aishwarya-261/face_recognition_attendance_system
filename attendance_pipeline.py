import cv2
import os
import pandas as pd
import numpy as np
import datetime
import time
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global Caches for Real-Time Performance
_mtcnn = None
_resnet = None
_trained_data = None

def get_models():
    """Singleton to load and cache models for real-time performance."""
    global _mtcnn, _resnet
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=device)
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _mtcnn, _resnet

def get_trained_data():
    """Singleton to load and cache trained face embeddings."""
    global _trained_data
    if _trained_data is None:
        try:
            with open("TrainingImageLabel/Trainer.pkl", 'rb') as f:
                _trained_data = pickle.load(f)
        except FileNotFoundError:
            return None
    return _trained_data

def ensure_folders():
    """Utility to ensure all required data folders exist."""
    for f in ['TrainingImage', 'TrainingImageLabel', 'StudentDetails', 'Attendance']:
        if not os.path.exists(f):
            os.makedirs(f)

def wipe_all_data():
    """Wipes all student images, records, and attendance logs."""
    import shutil
    ensure_folders() # Ensure they exist before trying to delete just in case
    for folder in ['TrainingImage', 'TrainingImageLabel', 'StudentDetails', 'Attendance']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    # Recreate empty folders
    ensure_folders()
    return "All data successfully cleared. You can start fresh now."

def enroll_from_image(enrollment_id, name, image, sample_num):
    """Enroll a student using a single image (generates exactly 20 augmented face crops)."""
    ensure_folders()
    try:
        enroll_id_int = int(enrollment_id)
    except Exception:
        return "Error: Enrollment ID must be numeric."
        
    if not name.strip():
        return "Error: Name cannot be empty."
        
    # Convert PIL to RGB
    pil_img = image.convert('RGB')
    
    # NEW: Detect and Crop Face first (Matches Image 3)
    mtcnn, _ = get_models()
    boxes, _ = mtcnn.detect(pil_img)
    
    if boxes is not None:
        # Take the most prominent face
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        # Add slight padding (20%)
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - int(w*0.1))
        y1 = max(0, y1 - int(h*0.1))
        x2 = min(pil_img.width, x2 + int(w*0.1))
        y2 = min(pil_img.height, y2 + int(h*0.1))
        pil_img = pil_img.crop((x1, y1, x2, y2))
    else:
        return "Error: No face detected in enrollment photo. Please look at the camera."

    # Augmentation pipeline: Flip, Zoom (scale), and Brightness (jitter)
    # REMOVED: degrees=15 (Matches Image 3)
    saving_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), translate=(0.05, 0.05)),
        transforms.Resize((160, 160)) # Ensure consistent size
    ])
    
    saved_count = 0
    # Generate and save exactly 20 samples
    for i in range(1, 21):
        try:
            filename = f"TrainingImage/{name}.{enrollment_id}.{i}.jpg"
            # Apply augmentation (except for the first one, which is the clean crop)
            aug_img = saving_trans(pil_img) if i > 1 else pil_img.resize((160, 160))
            aug_img.save(filename, "JPEG", quality=95)
            saved_count += 1
        except Exception as e:
            print(f"Failed to save sample {i}: {e}")
            
    # Update Record CSV
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        pd.DataFrame([[enrollment_id, name]], columns=['Id', 'Name']).to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)
        if enrollment_id not in df['Id'].values:
            pd.DataFrame([[enrollment_id, name]], columns=['Id', 'Name']).to_csv(df_path, mode='a', header=False, index=False)

    if saved_count >= 20:
        return f"Success! Generated exactly {saved_count} face-cropped samples for {name}."
    else:
        return f"Warning: Only saved {saved_count}/20 samples. Check server logs."

def capture_images(enrollment_id, name):
    # (Original desktop Capture code remains if needed for local use)
    ensure_folders()
    ...
    try:
        enroll_id_int = int(enrollment_id)
    except ValueError:
        return "Error: Enrollment ID must be numeric."
        
    if not name.isalpha():
        return "Error: Name must be alphabetical."
        
    mtcnn = MTCNN(keep_all=False, device=device)
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        return "Error: Could not open camera. Please check if it is connected or in use."
        
    sampleNum = 0
    window_name = 'Face Capturing - Wait & Look at Camera'
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # MTCNN works with RGB images
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is not None:
            # For enrollment, focus on the largest detected face (presumably the student)
            best_box_idx = 0
            if len(boxes) > 1:
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
                best_box_idx = np.argmax(areas)
                
            box = boxes[best_box_idx]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Make sure bounds are within frame
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    sampleNum += 1
                    filename = f"TrainingImage/{name}.{enrollment_id}.{sampleNum}.jpg"
                    cv2.imwrite(filename, face_img)
                
        cv2.imshow(window_name, frame)
        
        # Give the window a moment to initialize on first display
        if sampleNum == 0:
            cv2.waitKey(1)

        # Check if window is still open
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum >= 20:
            break
            
    cam.release()
    cv2.destroyAllWindows()
    
    # Save augmented images to disk physically
    saving_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    ])
    for idx in range(1, sampleNum + 1):
        filepath = f"TrainingImage/{name}.{enrollment_id}.{idx}.jpg"
        if os.path.exists(filepath):
            pil_img = Image.open(filepath).convert('RGB')
            # Apply exactly one random augmentation sequence and overwrite
            aug_img = saving_trans(pil_img)
            aug_img.save(filepath)
    
    row = [enroll_id_int, name]
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        df = pd.DataFrame(columns=['Id', 'Name'])
        df.to_csv(df_path, index=False)
        
    df = pd.read_csv(df_path)
    df = df.dropna(subset=['Id'])
    # Ensure ID is treated as integer column
    df['Id'] = df['Id'].astype(int)
    
    if enroll_id_int not in df['Id'].values:
        new_row = pd.DataFrame([row], columns=['Id', 'Name'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(df_path, index=False)
    
    return f"Success: Captured exactly {sampleNum} base images, fully augmented."

def train_images():
    ensure_folders()
    path = "TrainingImage"
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    if not imagePaths:
        return "No images found for training."
        
    # Load InceptionResnetV1 locally evaluated state
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    
    embeddings = []
    ids = []
    
    for imagePath in imagePaths:
        try:
            filename = os.path.split(imagePath)[-1]
            Id = int(filename.split('.')[1])
            
            img = Image.open(imagePath).convert('RGB')
            img_tensor = trans(img).unsqueeze(0).to(device)
            # 🌟 Standardize input for FaceNet
            img_tensor = (img_tensor - 127.5) / 128.0
            
            with torch.no_grad():
                emb = resnet(img_tensor).cpu().numpy()[0]
                # 🌟 L2 Normalize to unit length for hyper-sphere comparison
                emb = emb / (np.linalg.norm(emb) + 1e-6)
                
            embeddings.append(emb)
            ids.append(Id)
                
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
            continue
            
    # Save the embeddings array to disk
    model_data = {"embeddings": np.array(embeddings), "ids": np.array(ids)}
    with open("TrainingImageLabel/Trainer.pkl", 'wb') as f:
        pickle.dump(model_data, f)
        
    return f"Model Trained Successfully with High-Accuracy Logic ({len(imagePaths)} images)."

from PIL import ImageDraw, ImageFont

def recognize_face(image):
    """Recognize a face from a single image and return an annotated image (Optimized)."""
    ensure_folders()
    model_data = get_trained_data()
    if model_data is None:
        return "Error", "Model not found. Please train images first.", image
        
    saved_embeddings = model_data["embeddings"]
    saved_ids = model_data["ids"]
        
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        return "Error", "Student details not found.", image
    df = pd.read_csv(df_path)
    df = df.dropna(subset=['Id'])
    df['Id'] = df['Id'].astype(int)
    
    # Use cached models
    mtcnn, resnet = get_models()
    
    # Generate annotated image
    img_rgb = image.convert('RGB')
    draw = ImageDraw.Draw(img_rgb)
    boxes, probs = mtcnn.detect(img_rgb)
    
    if boxes is None:
        return "No Face", "No face detected in the image.", img_rgb
        
    results_list = []
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=5)
        
        try:
            face_crop = img_rgb.crop((x1, y1, x2, y2))
            img_tensor = trans(face_crop).unsqueeze(0).to(device)
            # 🌟 Standardize the capture exactly like training
            img_tensor = (img_tensor - 127.5) / 128.0
            
            with torch.no_grad():
                emb = resnet(img_tensor).cpu().numpy()[0]
                # 🌟 L2 Normalize for hyper-sphere compare
                emb = emb / (np.linalg.norm(emb) + 1e-6)
            
            # 🌟 GROUP-MEAN MATCHING: Calculate distance to each student cluster
            best_id = -1
            min_mean_dist = 2.0 # Unit vector dist range is [0, 2]
            
            for uid in np.unique(saved_ids):
                # Calculate the average distance to this student's 20 images
                student_masks = (saved_ids == uid)
                student_embs = saved_embeddings[student_masks]
                dists = np.linalg.norm(student_embs - emb, axis=1)
                mean_dist = np.mean(dists)
                
                if mean_dist < min_mean_dist:
                    min_mean_dist = mean_dist
                    best_id = uid
            
            # 🌟 TIGHT THRESHOLD: Standard 0.85 for FaceNet L2-Norm
            if min_mean_dist < 0.85:
                # Find Name
                name_row = df.loc[df['Id'] == best_id]['Name'].values
                name = name_row[0] if len(name_row) > 0 else "Unknown"
                display_text = f"{best_id}-{name}"
                draw.text((x1, y1 - 40), display_text, fill="lime")
                results_list.append(display_text)
            else:
                draw.text((x1, y1 - 40), "Unknown", fill="red")
        except Exception as e:
            print(f"Recognition Logic Error: {e}")
            continue
            
    final_status = "Recognized" if results_list else "No Match"
    return final_status, ", ".join(results_list), img_rgb

def log_attendance(display_text):
    """Safely log attendance to CSV (ID-Name format)."""
    try:
        parts = display_text.split("-")
        if len(parts) < 2: return
        Id = int(parts[0])
        name = parts[1]
        
        # Correct timing to IST (GMT+5:30)
        ist_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        date = ist_now.strftime('%Y-%m-%d')
        timeStamp = ist_now.strftime('%H:%M:%S')
        
        attendance_row = [Id, name, date, timeStamp]
        ensure_folders()
        fileName = "Attendance/Master_Attendance.csv"
        
        # Append to master log
        df = pd.DataFrame([attendance_row], columns=['Id', 'Name', 'Date', 'Time'])
        if os.path.exists(fileName):
            df.to_csv(fileName, mode='a', header=False, index=False)
        else:
            df.to_csv(fileName, mode='w', header=True, index=False)
        return True
    except Exception as e:
        print(f"Logging error: {e}")
        return False

def automatic_attendance():
    ensure_folders()
    try:
        with open("TrainingImageLabel/Trainer.pkl", 'rb') as f:
            model_data = pickle.load(f)
        saved_embeddings = model_data["embeddings"]
        saved_ids = model_data["ids"]
    except FileNotFoundError:
        return "Model not found. Please train images first."
        
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        return "Student details not found."
    df = pd.read_csv(df_path)
    df = df.dropna(subset=['Id'])
    df['Id'] = df['Id'].astype(int)
    
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        return "Error: Could not open camera. Please check if it is connected or in use."

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    first_frame = True
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                
                h_f, w_f = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Crop and format for ResNet
                try:
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    img_tensor = trans(face_crop).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = resnet(img_tensor).cpu().numpy()[0]
                    
                    # Compute L2 Euclidean distances
                    dists = np.linalg.norm(saved_embeddings - emb, axis=1)
                    min_dist_idx = np.argmin(dists)
                    min_dist = dists[min_dist_idx]
                    
                    # Distance < 1.2 is safer for FaceNet, especially with augmented data
                    if min_dist < 1.2:
                        Id = saved_ids[min_dist_idx]
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        name_row = df.loc[df['Id'] == Id]['Name'].values
                        name = name_row[0] if len(name_row) > 0 else "Unknown"
                        
                        tt = f"{Id}-{name}"
                        attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                        color = (0, 255, 0)
                    else:
                        Id = 'Unknown'
                        tt = str(Id)
                        color = (0, 0, 255)
                        
                    cv2.putText(frame, str(tt), (x1, y1 - 10), font, 1, color, 2)
                    
                except Exception as e:
                    print(e)
                    continue

        window_name = 'Taking Attendance (Press Q to finish)'
        cv2.imshow(window_name, frame)
        
        # To avoid OpenCV window bug closing instantly on the first frame
        if first_frame:
            cv2.waitKey(1)
            first_frame = False
            
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

        # Check if window is still open
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
    attendance = attendance[attendance['Id'] != 'Unknown']
    
    if len(attendance) > 0:
        # Create a new session folder
        ts_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join("Attendance", f"Session_{ts_folder}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        fileName = os.path.join(dir_path, "Attendance_Report.csv")
        attendance.to_csv(fileName, mode='w', header=True, index=False)
        
        # Also append to a master log for convenience
        master_file = "Attendance/Master_Attendance.csv"
        if os.path.exists(master_file):
            attendance.to_csv(master_file, mode='a', header=False, index=False)
        else:
            attendance.to_csv(master_file, mode='w', header=True, index=False)
            
        msg = f"Attendance saved to {fileName}"
    else:
         msg = "No recognized faces to mark attendance."
         
    cam.release()
    cv2.destroyAllWindows()
    return msg
