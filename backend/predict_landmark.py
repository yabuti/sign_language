#!/usr/bin/env python3
"""
Ethiopian Sign Language Prediction using MediaPipe Hand Landmarks
Uses TWO hands (126 features = 2 hands x 21 landmarks x 3 coordinates)
"""

import sys
import json
import os
import numpy as np
import cv2
from tensorflow import keras

# Try to import mediapipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print(json.dumps({"error": "MediaPipe not installed. Run: pip install mediapipe", "success": False}))
    sys.exit(1)

# ===== Configuration =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "eth_landmark_model.keras")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "eth_landmark_labels.json")

# Load labels
def load_labels():
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        # Convert to list ordered by index
        labels = [labels_dict[str(i)] for i in range(len(labels_dict))]
        return labels
    except Exception as e:
        return None

ETH_LABELS = load_labels()
if ETH_LABELS is None:
    ETH_LABELS = [
        'ሂድ', 'ህመም', 'መንደር', 'ምግብ', 'ሰላም', 'ቀለም', 'አመሰግናለሁ', 'አቁም', 'አዎን', 'እባክህ',
        'እንደገና', 'እገዛ', 'እግር', 'ውሃ', 'ይቅርታ', 'ድምፅ', 'ድንጋይ', 'ግራ', 'ጥሩ', 'ጨምር'
    ]

def load_model():
    """Load the landmark model"""
    try:
        if not os.path.exists(MODEL_PATH):
            return None, f"Model not found: {MODEL_PATH}"
        model = keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"


def extract_landmarks_from_frame(frame, hands):
    """
    Extract hand landmarks from a frame.
    Returns 126 features (2 hands x 21 landmarks x 3 coords) or None if no hands.
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        # Initialize landmarks for 2 hands (filled with zeros)
        all_landmarks = np.zeros(126)
        
        # Get handedness info
        for idx, (hand_landmarks, handedness) in enumerate(zip(
            results.multi_hand_landmarks[:2],
            results.multi_handedness[:2]
        )):
            hand_label = handedness.classification[0].label
            hand_offset = 0 if hand_label == "Left" else 63
            
            for i, lm in enumerate(hand_landmarks.landmark):
                all_landmarks[hand_offset + i*3] = lm.x
                all_landmarks[hand_offset + i*3 + 1] = lm.y
                all_landmarks[hand_offset + i*3 + 2] = lm.z
        
        return all_landmarks
    
    return None


def process_video(video_path, model):
    """Process video and predict using landmarks"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video", "success": False}
        
        # Read all frames
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        
        if len(all_frames) == 0:
            return {"error": "No frames in video", "success": False}
        
        # Select frames to process (7 frames spread across video)
        total_frames = len(all_frames)
        if total_frames >= 7:
            percentages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            frame_indices = [min(int(total_frames * p), total_frames - 1) for p in percentages]
        else:
            frame_indices = list(range(total_frames))
        
        frames = [all_frames[i] for i in frame_indices]
        
        # Extract landmarks from frames
        all_landmarks = []
        
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:
            for frame in frames:
                # Rotate if landscape (Iriun webcam fix)
                if frame.shape[1] > frame.shape[0]:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                landmarks = extract_landmarks_from_frame(frame, hands)
                if landmarks is not None:
                    all_landmarks.append(landmarks)
        
        if len(all_landmarks) == 0:
            return {
                "error": "No hands detected in video. Please show both hands clearly.",
                "success": False
            }
        
        # Predict on all frames with landmarks
        landmarks_array = np.array(all_landmarks)
        predictions = model.predict(landmarks_array, verbose=0)
        
        # Find best prediction (highest confidence)
        best_idx = 0
        best_conf = 0
        for i, pred in enumerate(predictions):
            max_conf = float(np.max(pred))
            if max_conf > best_conf:
                best_conf = max_conf
                best_idx = i
        
        best_pred = predictions[best_idx]
        pred_class = int(np.argmax(best_pred))
        confidence = float(best_pred[pred_class])
        
        # Get label
        if pred_class < len(ETH_LABELS):
            predicted_label = ETH_LABELS[pred_class]
        else:
            predicted_label = f"Sign_{pred_class}"
        
        # Top 3
        top_3_indices = np.argsort(best_pred)[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            label = ETH_LABELS[idx] if idx < len(ETH_LABELS) else f"Sign_{idx}"
            top_3.append({
                "label": label,
                "confidence": round(float(best_pred[idx]) * 100, 2)
            })
        
        # Voting for stability check
        from collections import Counter
        all_pred_classes = [int(np.argmax(p)) for p in predictions]
        vote_counts = Counter(all_pred_classes)
        most_common, most_count = vote_counts.most_common(1)[0]
        agreement = most_count / len(predictions)
        
        return {
            "text": predicted_label,
            "confidence": round(confidence * 100, 2),
            "top_3": top_3,
            "model_used": "Ethiopian Sign Language (Landmark)",
            "frames_with_hands": len(all_landmarks),
            "total_frames": len(frames),
            "agreement": f"{most_count}/{len(predictions)} ({agreement*100:.0f}%)",
            "is_stable": bool(agreement >= 0.5),
            "success": True
        }
        
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}", "success": False}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No video path provided", "success": False}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(json.dumps({"error": f"Video not found: {video_path}", "success": False}))
        sys.exit(1)
    
    # Load model
    model, error = load_model()
    if error:
        print(json.dumps({"error": error, "success": False}))
        sys.exit(1)
    
    # Process video
    result = process_video(video_path, model)
    
    # Output JSON
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
