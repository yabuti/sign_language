"""
Compare the debug frame from webcam with training images
to see why predictions are wrong
"""
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# Load model
print("Loading model...")
model = tf.keras.models.load_model('eth_model_robust.keras')

labels = ['ሂድ', 'ህመም', 'መንደር', 'ምግብ', 'ሰላም', 'ቀለም', 'አመሰግናለሁ', 'አቁም', 'አዎን', 'እባክህ', 'እንደገና', 'እገዛ', 'እግር', 'ውሃ', 'ይቅርታ', 'ድምፅ', 'ድንጋይ', 'ግራ', 'ጥሩ', 'ጨምር']

# Test the debug frame from webcam
debug_path = 'backend/debug_frame.jpg'

print(f"\n=== Testing debug frame: {debug_path} ===")

# Read debug frame (it's already 224x224 or combined frames)
img = cv2.imread(debug_path)
if img is None:
    print("Could not read debug_frame.jpg - do a webcam capture first!")
    exit()

print(f"Debug frame shape: {img.shape}")

# If it's combined frames (wider than 224), split and test each
if img.shape[1] > 300:  # Combined frames
    num_frames = img.shape[1] // 224
    print(f"Found {num_frames} combined frames, testing each...")
    
    for i in range(num_frames):
        frame = img[:, i*224:(i+1)*224]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_norm = frame_rgb.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_norm, axis=0)
        
        pred = model.predict(frame_batch, verbose=0)
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        print(f"  Frame {i+1}: {labels[idx]} ({conf:.1f}%)")
        
        # Show top 3
        top3 = np.argsort(pred[0])[-3:][::-1]
        print(f"    Top 3: {[(labels[j], f'{pred[0][j]*100:.1f}%') for j in top3]}")
else:
    # Single frame
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    frame_batch = np.expand_dims(frame_norm, axis=0)
    
    pred = model.predict(frame_batch, verbose=0)
    idx = np.argmax(pred[0])
    conf = pred[0][idx] * 100
    
    print(f"Prediction: {labels[idx]} ({conf:.1f}%)")
    
    # Show top 3
    top3 = np.argsort(pred[0])[-3:][::-1]
    print(f"Top 3: {[(labels[j], f'{pred[0][j]*100:.1f}%') for j in top3]}")

print("\n=== Compare with training images ===")
print("Open backend/debug_frame.jpg and compare with eth_frames/train/[class]/ images")
print("Check: rotation, background, lighting, hand position")
