#!/usr/bin/env python3
"""
Video/Webcam ASL Prediction using LSTM model
Processes video frames and extracts pose landmarks for prediction
"""

import sys
import json
import os
import numpy as np
import cv2
from tensorflow import keras
import mediapipe as mp

# ===== Configuration =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "asl_model_lstm.h5")
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225  # 75 landmarks * 3 coordinates (x, y, z)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_model():
    """Load the LSTM model"""
    try:
        if not os.path.exists(MODEL_PATH):
            print(json.dumps({"error": f"Model not found at {MODEL_PATH}"}))
            sys.exit(1)
        
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

def extract_keypoints(results):
    """Extract pose, face, and hand landmarks"""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Combine all landmarks (total: 33+468+21+21 = 543 landmarks * 3 = 1629 features)
    # But model expects 225 features, so we'll use only pose and hands
    keypoints = np.concatenate([pose, lh, rh])  # 33*3 + 21*3 + 21*3 = 225
    
    return keypoints

def process_video(video_path, model):
    """Process video and extract sequence of keypoints"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Failed to open video file", "success": False}
        
        sequence = []
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames evenly to get exactly SEQUENCE_LENGTH frames
            frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
            
            while cap.isOpened() and len(sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Only process frames at specified indices
                if frame_count in frame_indices:
                    # Convert to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    
                    # Make detection
                    results = holistic.process(image)
                    
                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                
                frame_count += 1
            
            cap.release()
        
        # Pad sequence if needed
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append(np.zeros(NUM_FEATURES))
        
        # Convert to numpy array
        sequence = np.array(sequence)
        
        # Ensure correct shape
        if sequence.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
            return {
                "error": f"Invalid sequence shape: {sequence.shape}, expected ({SEQUENCE_LENGTH}, {NUM_FEATURES})",
                "success": False
            }
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        predictions = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class": int(idx),
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return {
            "text": f"Sign_{predicted_index}",  # Replace with actual labels if available
            "class": int(predicted_index),
            "confidence": round(confidence * 100, 2),
            "top_3": top_3_predictions,
            "success": True
        }
        
    except Exception as e:
        return {
            "text": "Error processing video",
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No video path provided"}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(json.dumps({"error": f"Video not found: {video_path}"}))
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Process video
    result = process_video(video_path, model)
    
    # Output JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
