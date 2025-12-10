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
    "eth_mobilenet": os.path.join(os.path.dirname(__file__), "..", "eth_model_robust.keras"),
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


def normalize_lighting_for_asl(frame):
    """
    Normalize lighting for ASL model using CLAHE.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)
    lab_normalized = cv2.merge([l_normalized, a, b])
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)


def extract_mobilenet_features(frame):
    """Extract MobileNetV2 features from a single frame"""
    extractor = get_feature_extractor()
    
    # Normalize lighting first
    frame = normalize_lighting_for_asl(frame)
    
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


def normalize_lighting(frame):
    """
    Normalize lighting conditions to make model robust to different environments.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for better results.
    """
    # Convert to LAB color space (L = lightness, A/B = color)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)
    
    # Merge back
    lab_normalized = cv2.merge([l_normalized, a, b])
    frame_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    
    return frame_normalized


def process_video_ethiopian(video_path, model):
    """
    Process video for Ethiopian MobileNet model.
    Takes 7 frames, uses voting + averaging for stable prediction.
    Requires majority agreement for confident results.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video", "success": False}
        
        # Read ALL frames first (webm seeking doesn't work well)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        cap.release()
        
        total_frames = len(all_frames)
        if total_frames == 0:
            return {"error": "Failed to read frames from video (empty)", "success": False}
        
        duration_seconds = total_frames / fps
        
        # Select 7 frames: evenly spread across the video (more frames = more stable)
        NUM_FRAMES = 7
        if total_frames >= NUM_FRAMES:
            # Take frames at 20%, 30%, 40%, 50%, 60%, 70%, 80% of the video
            percentages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            frame_indices = [min(int(total_frames * p), total_frames - 1) for p in percentages]
        else:
            # Use all available frames
            frame_indices = list(range(total_frames))
        
        # Get the selected frames
        frames = [all_frames[i] for i in frame_indices]
        frames_info = [f"{i}" for i in frame_indices]
        
        if len(frames) == 0:
            return {"error": "Failed to read frames from video", "success": False}
        
        # Process all frames first (preprocessing)
        processed_frames = []
        batch_input = []
        
        labels = ETH_LABELS
        
        for frame in frames:
            # Rotate frame 90 degrees clockwise if it's landscape
            if frame.shape[1] > frame.shape[0]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Preprocess for Ethiopian model (224x224)
            frame_resized = cv2.resize(frame, (224, 224))
            processed_frames.append(frame_resized)
            
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            batch_input.append(frame_normalized)
        
        # BATCH PREDICTION - much faster than predicting one by one!
        batch_array = np.array(batch_input)  # Shape: (7, 224, 224, 3)
        all_predictions = model.predict(batch_array, verbose=0)  # Single prediction call
        individual_pred_indices = [np.argmax(pred) for pred in all_predictions]
        
        # Save debug image: combine frames side by side (max 5 for display)
        debug_path = os.path.join(os.path.dirname(__file__), "debug_frame.jpg")
        display_frames = processed_frames[:5] if len(processed_frames) > 5 else processed_frames
        if len(display_frames) > 1:
            combined = np.hstack(display_frames)
            cv2.imwrite(debug_path, combined)
        elif len(display_frames) == 1:
            cv2.imwrite(debug_path, display_frames[0])
        
        # === FIND BEST PREDICTION: Use highest confidence frame ===
        # Instead of voting, find the frame with highest confidence
        best_frame_idx = 0
        best_confidence = 0
        best_class_idx = 0
        
        for i, pred in enumerate(all_predictions):
            max_conf = float(np.max(pred))
            if max_conf > best_confidence:
                best_confidence = max_conf
                best_frame_idx = i
                best_class_idx = np.argmax(pred)
        
        # Also check voting for stability info
        from collections import Counter
        vote_counts = Counter(individual_pred_indices)
        most_common_idx, most_common_count = vote_counts.most_common(1)[0]
        agreement_pct = (most_common_count / len(frames)) * 100
        
        # Get individual frame predictions for debugging
        individual_preds = []
        for i, pred in enumerate(all_predictions):
            idx = np.argmax(pred)
            label = labels[idx] if idx < len(labels) else f"Sign_{idx}"
            conf = float(pred[idx]) * 100
            marker = " ★" if i == best_frame_idx else ""
            individual_preds.append(f"{label}({conf:.0f}%){marker}")
        
        # Use the BEST frame's prediction (highest confidence)
        if best_class_idx < len(labels):
            predicted_label = labels[best_class_idx]
        else:
            predicted_label = f"Sign_{best_class_idx}"
        
        # === STABILITY CHECK ===
        # Check if best prediction matches majority vote
        is_stable = (best_class_idx == most_common_idx) or (best_confidence > 0.7)
        
        if is_stable:
            stability_warning = f"✓ Best frame {best_frame_idx+1} ({best_confidence*100:.0f}%)"
        else:
            stability_warning = f"⚠️ Mixed results - best: {predicted_label}, majority: {labels[most_common_idx]}"
        
        # Top 3 from best frame
        best_pred = all_predictions[best_frame_idx]
        top_3_indices = np.argsort(best_pred)[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            label = labels[idx] if idx < len(labels) else f"Sign_{idx}"
            top_3.append({
                "label": label,
                "confidence": round(float(best_pred[idx]) * 100, 2)
            })
        
        final_confidence = best_confidence * 100
        
        return {
            "text": predicted_label,
            "confidence": round(final_confidence, 2),
            "top_3": top_3,
            "model_used": MODEL_NAMES["eth_mobilenet"],
            "frame_info": f"{stability_warning} | Frames: {', '.join(frames_info)} | Preds: {', '.join(individual_preds[:5])}{'...' if len(individual_preds) > 5 else ''}",
            "debug_image": "backend/debug_frame.jpg",
            "frames_used": len(frames),
            "agreement": f"{most_common_count}/{len(frames)} ({agreement_pct:.0f}%)",
            "is_stable": bool(is_stable),
            "success": True
        }
        
    except Exception as e:
        import traceback
        return {"error": f"{str(e)} - {traceback.format_exc()}", "success": False}

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
