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

def ensure_folders():
    for f in ['TrainingImage', 'TrainingImageLabel', 'StudentDetails', 'Attendance']:
        if not os.path.exists(f):
            os.makedirs(f)

def augment_image_dynamic(pil_img):
    """Apply random transforms to a PIL image for dynamic data augmentation."""
    augment_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    return augment_trans(pil_img)

def enroll_from_image(enrollment_id, name, image, sample_num):
    """Enroll a student using a single image (for web/gradio)."""
    ensure_folders()
    try:
        enroll_id_int = int(enrollment_id)
    except ValueError:
        return "Error: Enrollment ID must be numeric."
        
    if not name.replace(" ", "").isalpha():
        return "Error: Name must be alphabetical."
        
    # Convert PIL to BGR for augmentation functions
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Save the base images
    filename = f"TrainingImage/{name}.{enrollment_id}.{sample_num}.jpg"
    cv2.imwrite(filename, img_cv)
    
    # Update CSV record
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        df = pd.DataFrame(columns=['Id', 'Name'])
    else:
        df = pd.read_csv(df_path)
    
    df['Id'] = df['Id'].astype(int)
    if enroll_id_int not in df['Id'].values:
        new_row = pd.DataFrame([[enroll_id_int, name]], columns=['Id', 'Name'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(df_path, index=False)
        
    return f"Success: Images for {name} (ID: {enrollment_id}) added."

def capture_images(enrollment_id, name):
    ensure_folders()
    
    # Make enrollment_id an integer early
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
            
            # Encode image and append to training arrays
            img_tensor_orig = trans(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb_orig = resnet(img_tensor_orig).cpu().numpy()[0]
            embeddings.append(emb_orig)
            ids.append(Id)
                
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
            continue
            
    # Save the embeddings array to disk
    model_data = {"embeddings": np.array(embeddings), "ids": np.array(ids)}
    with open("TrainingImageLabel/Trainer.pkl", 'wb') as f:
        pickle.dump(model_data, f)
        
    return f"Model Trained Successfully. Processed {len(imagePaths)} images (including augmented variations)."

def recognize_face(image):
    """Recognize a face from a single image (for web/gradio)."""
    ensure_folders()
    try:
        with open("TrainingImageLabel/Trainer.pkl", 'rb') as f:
            model_data = pickle.load(f)
        saved_embeddings = model_data["embeddings"]
        saved_ids = model_data["ids"]
    except FileNotFoundError:
        return "Error", "Model not found. Please train images first."
        
    df_path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(df_path):
        return "Error", "Student details not found."
    df = pd.read_csv(df_path)
    df = df.dropna(subset=['Id'])
    df['Id'] = df['Id'].astype(int)
    
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Convert image to RGB
    img_rgb = image.convert('RGB')
    boxes, probs = mtcnn.detect(img_rgb)
    
    if boxes is None:
        return "No Face", "No face detected in the image."
        
    results = []
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        try:
            face_crop = img_rgb.crop((x1, y1, x2, y2))
            img_tensor = trans(face_crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = resnet(img_tensor).cpu().numpy()[0]
            
            dists = np.linalg.norm(saved_embeddings - emb, axis=1)
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            
            if min_dist < 1.2:
                Id = saved_ids[min_dist_idx]
                name_row = df.loc[df['Id'] == Id]['Name'].values
                name = name_row[0] if len(name_row) > 0 else "Unknown"
                
                # Mark Attendance
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                attendance_row = [Id, name, date, timeStamp]
                fileName = "Attendance/Master_Attendance.csv"
                if os.path.exists(fileName):
                    pd.DataFrame([attendance_row]).to_csv(fileName, mode='a', header=False, index=False)
                else:
                    pd.DataFrame([attendance_row], columns=['Id', 'Name', 'Date', 'Time']).to_csv(fileName, mode='w', header=True, index=False)
                    
                results.append(f"{name} (ID: {Id})")
            else:
                results.append("Unknown")
        except Exception as e:
            continue
            
    return "Recognized", ", ".join(results)

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
