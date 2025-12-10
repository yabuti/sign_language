#!/usr/bin/env python3
"""
Sign Language Image Prediction Script
Uses sign_language_model.h5 for ASL alphabet (A-Z)
"""

import sys
import json
import os
import numpy as np
from tensorflow import keras
from PIL import Image

# ===== Configuration =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "sign_language_model.h5")

# ASL Alphabet Labels (26 letters A-Z)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_model():
    """Load the Keras model"""
    try:
        if not os.path.exists(MODEL_PATH):
            return None, f"Model not found at {MODEL_PATH}"
        
        model = keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

def preprocess_image(image_path, target_size=(64, 64)):
    """Preprocess image for prediction"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to model's expected input size (64x64)
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    except Exception as e:
        return None, f"Failed to preprocess image: {str(e)}"

def predict_image(image_path, model):
    """Predict ASL sign from image"""
    try:
        # Preprocess image
        img_array, error = preprocess_image(image_path)
        if error:
            return {"text": error, "confidence": 0.0, "success": False}
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get label
        if predicted_index < len(LABELS):
            predicted_label = LABELS[predicted_index]
        else:
            predicted_label = f"Letter_{predicted_index}"
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            label = LABELS[idx] if idx < len(LABELS) else f"Letter_{idx}"
            top_3.append({
                "label": label,
                "confidence": round(float(predictions[0][idx]) * 100, 2)
            })
        
        result_text = f"✅ {predicted_label} ({round(confidence * 100, 2)}%)\n\n"
        result_text += "Top 3 Predictions:\n"
        for i, pred in enumerate(top_3):
            result_text += f"{i+1}. {pred['label']} - {pred['confidence']}%\n"
        
        return {
            "text": result_text,
            "confidence": round(confidence * 100, 2),
            "top_3": top_3,
            "success": True
        }
        
    except Exception as e:
        return {
            "text": f"Error processing image: {str(e)}",
            "confidence": 0.0,
            "success": False
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided", "text": "No image provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}", "text": "Image not found"}))
        sys.exit(1)
    
    # Load model
    model, error = load_model()
    if error:
        print(json.dumps({"error": error, "text": error}))
        sys.exit(1)
    
    # Predict
    result = predict_image(image_path, model)
    
    # Output JSON
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
