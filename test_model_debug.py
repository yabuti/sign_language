import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path

# Load model
print("Loading model...")
model = tf.keras.models.load_model('eth_model_robust.keras')
print(f'Model loaded. Output shape: {model.output_shape}')

# Labels
labels = ['ሂድ', 'ህመም', 'መንደር', 'ምግብ', 'ሰላም', 'ቀለም', 'አመሰግናለሁ', 'አቁም', 'አዎን', 'እባክህ', 'እንደገና', 'እገዛ', 'እግር', 'ውሃ', 'ይቅርታ', 'ድምፅ', 'ድንጋይ', 'ግራ', 'ጥሩ', 'ጨምር']

# Get all class folders
train_path = Path('eth_frames/train')
class_folders = list(train_path.iterdir())

print(f"\nFound {len(class_folders)} classes")
print("Testing first image from each class...\n")

for folder in sorted(class_folders)[:5]:  # Test first 5 classes
    class_name = folder.name
    files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
    
    if files:
        img_path = str(files[0])
        
        # Read image using numpy to handle unicode paths
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Could not read: {class_name}")
            continue
            
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)
        
        pred = model.predict(img_batch, verbose=0)
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        correct = "OK" if labels[idx] == class_name else "WRONG"
        print(f'{correct}: {class_name} -> {labels[idx]} ({conf:.1f}%)')
