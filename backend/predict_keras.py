#!/usr/bin/env python3
"""
Keras/TensorFlow ASL Image Prediction Script
Uses sign_language_model.h5 for predictions
"""

import sys
import json
import os
import numpy as np
from tensorflow import keras
from PIL import Image

# ===== Configuration =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "sign_language_model.h5")
# Common ASL alphabet labels (A-Z)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_model():
    """Load the Keras model"""
    try:
        if not os.path.exists(MODEL_PATH):
            print(json.dumps({"error": f"Model not found at {MODEL_PATH}"}))
            sys.exit(1)
        
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

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
        
        return img_array
    except Exception as e:
        print(json.dumps({"error": f"Failed to preprocess image: {str(e)}"}))
        sys.exit(1)

def predict_image(image_path, model):
    """Predict ASL sign from image"""
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get label
        if predicted_index < len(LABELS):
            predicted_label = LABELS[predicted_index]
        else:
            predicted_label = f"Class_{predicted_index}"
        
        return {
            "text": predicted_label,
            "confidence": round(confidence * 100, 2),
            "success": True
        }
        
    except Exception as e:
        return {
            "text": "Error processing image",
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Predict
    result = predict_image(image_path, model)
    
    # Output JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
