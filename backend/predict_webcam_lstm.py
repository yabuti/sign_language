#!/usr/bin/env python3
"""
Real-time Webcam ASL Prediction using LSTM model
Captures frames from webcam and predicts sign language
"""

import sys
import json
import os
import numpy as np
import cv2
from tensorflow import keras
import mediapipe as mp
import base64

# ===== Configuration =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "asl_model_lstm.h5")
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def load_model():
    """Load the LSTM model"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

def extract_keypoints(results):
    """Extract pose and hand landmarks"""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    keypoints = np.concatenate([pose, lh, rh])
    return keypoints

def process_frame_data(frame_data, model):
    """Process base64 encoded frame data"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode frame", "success": False}
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            results = holistic.process(image)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        
        return {
            "keypoints": keypoints.tolist(),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def predict_sequence(sequence_data, model):
    """Predict from sequence of keypoints"""
    try:
        sequence = np.array(sequence_data)
        
        # Pad if needed
        if len(sequence) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(sequence), NUM_FEATURES))
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        predictions = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        return {
            "text": f"Sign_{predicted_index}",
            "class": int(predicted_index),
            "confidence": round(confidence * 100, 2),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No action specified"}))
        sys.exit(1)
    
    action = sys.argv[1]
    model = load_model()
    
    if action == "process_frame":
        # Process single frame
        frame_data = sys.stdin.read()
        result = process_frame_data(frame_data, model)
        print(json.dumps(result))
        
    elif action == "predict":
        # Predict from sequence
        sequence_data = json.loads(sys.stdin.read())
        result = predict_sequence(sequence_data, model)
        print(json.dumps(result))
    
    else:
        print(json.dumps({"error": f"Unknown action: {action}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
