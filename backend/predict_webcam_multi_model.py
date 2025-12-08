#!/usr/bin/env python3
"""
Multi-Model Webcam Sign Language Prediction
- ASL Model: CNN-LSTM with MobileNet features (30 frames × 1280 features)
- Ethiopian Model: MobileNet single image (224×224×3)
"""

import sys
import json
import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== Configuration =====
MODEL_PATHS = {
    "asl_cnn_lstm": os.path.join(os.path.dirname(__file__), "..", "asl_best_cnn_lstm.keras"),
    "eth_mobilenet": os.path.join(os.path.dirname(__file__), "..", "eth_model_mobilenet_best.keras"),
}

# ASL Labels (30 words - sorted alphabetically as they were during training)
ASL_LABELS = [
    'accident', 'basketball', 'bed', 'before', 'bowling', 'call', 'candy', 'change',
    'cold', 'computer', 'cool', 'corn', 'cousin', 'dark', 'drink', 'go', 'help',
    'last', 'later', 'man', 'pizza', 'shirt', 'short', 'tall', 'thanksgiving',
    'thin', 'trade', 'what', 'who', 'yes'
]

# Ethiopian Labels (20 words - exact order from train_generator.class_indices after removing 'split')
# Index 0 was 'split' which was removed, so indices shifted down by 1
# Original: split:0, ሂድ:1, ህመም:2, መንደር:3, ምግብ:4, ሰላም:5, ቀለም:6, አመሰግናለሁ:7, አቁም:8, አዎን:9, 
#           እባክህ:10, እንደገና:11, እገዛ:12, እግር:13, ውሃ:14, ይቅርታ:15, ድምፅ:16, ድንጋይ:17, ግራ:18, ጥሩ:19, ጨምር:20
# After removing split: ሂድ:0, ህመም:1, መንደር:2, ምግብ:3, ሰላም:4, ቀለም:5, አመሰግናለሁ:6, አቁም:7, አዎን:8,
#                       እባክህ:9, እንደገና:10, እገዛ:11, እግር:12, ውሃ:13, ይቅርታ:14, ድምፅ:15, ድንጋይ:16, ግራ:17, ጥሩ:18, ጨምር:19
ETH_LABELS = [
    'ሂድ',        # 0
    'ህመም',       # 1
    'መንደር',      # 2
    'ምግብ',       # 3
    'ሰላም',       # 4
    'ቀለም',       # 5
    'አመሰግናለሁ',   # 6
    'አቁም',       # 7
    'አዎን',       # 8
    'እባክህ',      # 9
    'እንደገና',     # 10
    'እገዛ',       # 11
    'እግር',       # 12
    'ውሃ',        # 13
    'ይቅርታ',      # 14
    'ድምፅ',       # 15
    'ድንጋይ',      # 16
    'ግራ',        # 17
    'ጥሩ',        # 18
    'ጨምር'       # 19
]

# Model display names
MODEL_NAMES = {
    "asl_cnn_lstm": "American Sign Language (CNN-LSTM)",
    "eth_mobilenet": "Ethiopian Sign Language (MobileNet)"
}

# Global feature extractor for ASL model
feature_extractor = None

def get_feature_extractor():
    """Get MobileNetV2 feature extractor for ASL model"""
    global feature_extractor
    if feature_extractor is None:
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        feature_extractor = base_model
    return feature_extractor

def load_model(model_name):
    """Load the specified model"""
    try:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            return None, f"Model not found: {model_name}"
        model = keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"


def extract_mobilenet_features(frame):
    """Extract MobileNetV2 features from a single frame"""
    extractor = get_feature_extractor()
    
    # Resize to 224x224
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Preprocess for MobileNet
    frame_preprocessed = preprocess_input(frame_rgb.astype(np.float32))
    frame_batch = np.expand_dims(frame_preprocessed, axis=0)
    
    # Extract features (1280-dimensional vector)
    features = extractor.predict(frame_batch, verbose=0)
    return features[0]

def process_video_asl(video_path, model):
    """Process video for ASL CNN-LSTM model (30 frames × 1280 features)"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video", "success": False}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return {"error": "Empty video file", "success": False}
        
        # Sample 30 frames evenly
        frame_indices = np.linspace(0, max(0, total_frames - 1), 30, dtype=int)
        
        features_sequence = []
        frame_count = 0
        
        while cap.isOpened() and len(features_sequence) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count in frame_indices:
                # Extract MobileNet features
                features = extract_mobilenet_features(frame)
                features_sequence.append(features)
            
            frame_count += 1
        
        cap.release()
        
        # Pad if needed
        while len(features_sequence) < 30:
            features_sequence.append(np.zeros(1280))
        
        # Convert to numpy array: (30, 1280)
        sequence = np.array(features_sequence)
        sequence = np.expand_dims(sequence, axis=0)  # (1, 30, 1280)
        
        # Predict
        predictions = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get label
        labels = ASL_LABELS
        if predicted_index < len(labels):
            predicted_label = labels[predicted_index]
        else:
            predicted_label = f"Sign_{predicted_index}"
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            label = labels[idx] if idx < len(labels) else f"Sign_{idx}"
            top_3.append({
                "label": label,
                "confidence": round(float(predictions[0][idx]) * 100, 2)
            })
        
        return {
            "text": predicted_label,
            "confidence": round(confidence * 100, 2),
            "top_3": top_3,
            "model_used": MODEL_NAMES["asl_cnn_lstm"],
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}


def process_video_ethiopian(video_path, model):
    """Process video for Ethiopian MobileNet model (single frame 224×224×3)"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video", "success": False}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try to read frames - sometimes frame count is wrong for webm
        frame = None
        
        # Try middle frame first
        if total_frames > 0:
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
        
        # If that failed, try reading from start
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(30):  # Try to get any frame from first 30
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
        
        cap.release()
        
        if frame is None:
            return {"error": "Failed to read frame from video", "success": False}
        
        # Rotate frame 90 degrees clockwise if it's landscape (wider than tall)
        # This fixes Iriun webcam orientation issue
        if frame.shape[1] > frame.shape[0]:  # width > height means landscape
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Note: Don't mirror - training videos were not mirrored
        # frame = cv2.flip(frame, 1)  # Uncomment if needed
        
        # Preprocess for Ethiopian model (224x224)
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Debug: Save the processed frame to see what model receives
        debug_path = os.path.join(os.path.dirname(__file__), "debug_frame.jpg")
        cv2.imwrite(debug_path, frame_resized)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)  # (1, 224, 224, 3)
        
        # Predict
        predictions = model.predict(frame_batch, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get label
        labels = ETH_LABELS
        if predicted_index < len(labels):
            predicted_label = labels[predicted_index]
        else:
            predicted_label = f"Sign_{predicted_index}"
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            label = labels[idx] if idx < len(labels) else f"Sign_{idx}"
            top_3.append({
                "label": label,
                "confidence": round(float(predictions[0][idx]) * 100, 2)
            })
        
        return {
            "text": predicted_label,
            "confidence": round(confidence * 100, 2),
            "top_3": top_3,
            "model_used": MODEL_NAMES["eth_mobilenet"],
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No video path provided", "success": False}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "asl_cnn_lstm"
    
    if not os.path.exists(video_path):
        print(json.dumps({"error": f"Video not found: {video_path}", "success": False}))
        sys.exit(1)
    
    # Load model
    model, error = load_model(model_name)
    if error:
        print(json.dumps({"error": error, "success": False}))
        sys.exit(1)
    
    # Process based on model type
    if model_name == "eth_mobilenet":
        result = process_video_ethiopian(video_path, model)
    else:
        result = process_video_asl(video_path, model)
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()
