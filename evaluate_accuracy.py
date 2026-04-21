import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model():
    print("Initializing models for accuracy evaluation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # We use InceptionResnetV1 just like the main pipeline
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    
    path = "TrainingImage"
    if not os.path.exists(path):
        print(f"Error: Folder '{path}' not found.")
        return
        
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    if len(imagePaths) < 10:
        print("Not enough images to perform a secure split test. Please capture at least 1 person's dataset (20 images).")
        return
        
    embeddings = []
    labels = []
    
    print(f"Extracting 512-D Facial Signatures for {len(imagePaths)} stored images...")
    for imagePath in imagePaths:
        try:
            filename = os.path.split(imagePath)[-1]
            parts = filename.split('.')
            if len(parts) >= 3:
                label_id = int(parts[1])
                
                # TrainingImages are already cropped during capture
                img = Image.open(imagePath).convert('RGB')
                img_tensor = trans(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb = resnet(img_tensor).cpu().numpy()[0]
                    
                embeddings.append(emb)
                labels.append(label_id)
        except Exception as e:
            continue
            
    if not embeddings:
        print("Failed to extract embeddings.")
        return
        
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # In case there's only 1 class or uneven counts, stratify might fail. 
    # Fallback to pure random split if stratify throws an error.
    try:
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.25, random_state=42, stratify=labels)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.25, random_state=42)
    
    print(f"\nCreated Split: {len(X_train)} Reference (Gallery) images | {len(X_test)} Testing (Query) images.")
    
    y_pred = []
    threshold = 1.2 # The exact same threshold we configured for real-time inference
    
    for test_emb in X_test:
        # Simulate real-time Euclidian calculations
        dists = np.linalg.norm(X_train - test_emb, axis=1)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist < threshold:
            y_pred.append(y_train[min_dist_idx])
        else:
            y_pred.append(-1) # 'Unknown' status
            
    # Calculate holistic performance
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"FACE RECOGNITION SYSTEM ACCURACY: {acc * 100:.2f}%")
    print("="*50)
    
    print("\nDetailed Metrics (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("*(Class -1 defines faces incorrectly identified as 'Unknown')*\n")

if __name__ == "__main__":
    evaluate_model()
