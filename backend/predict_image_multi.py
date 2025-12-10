#!/usr/bin/env python3
"""
Multi-Model Image Prediction Script
- ASL Alphabet: sign_language_model.h5 (A-Z, 64x64)
- Ethiopian: eth_model_mobilenet_best.keras (20 signs, 224x224)
"""

import sys
import json
import os
import numpy as np
from tensorflow import keras
from PIL import Image

# ===== Model Paths =====
MODEL_PATHS = {
    "asl_alphabet": os.path.join(os.path.dirname(__file__), "..", "frontend", "sign_language_model.h5"),
    "ethiopian": os.path.join(os.path.dirname(__file__), "..", "eth_model_mobilenet_best.keras"),
}

# ===== Labels =====
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

ETH_LABELS = [
    'ሂድ',        # 0 - Go
    'ህመም',       # 1 - Pain
    'መንደር',      # 2 - Village
    'ምግብ',       # 3 - Food
    'ሰላም',       # 4 - Hello/Peace
    'ቀለም',       # 5 - Color
    'አመሰግናለሁ',   # 6 - Thank you
    'አቁም',       # 7 - Stop
    'አዎን',       # 8 - Yes
    'እባክህ',      # 9 - Please
    'እንደገና',     # 10 - Again
    'እገዛ',       # 11 - Help
    'እግር',       # 12 - Foot/Leg
    'ውሃ',        # 13 - Water
    'ይቅርታ',      # 14 - Sorry
    'ድምፅ',       # 15 - Sound/Voice
    'ድንጋይ',      # 16 - Stone
    'ግራ',        # 17 - Left
    'ጥሩ',        # 18 - Good
    'ጨምር'       # 19 - Add
]

# Model configs
MODEL_CONFIG = {
    "asl_alphabet": {
        "labels": ASL_LABELS,
        "input_size": (64, 64),
        "name": "ASL Alphabet (A-Z)"
    },
    "ethiopian": {
        "labels": ETH_LABELS,
        "input_size": (224, 224),
        "name": "Ethiopian Sign Language"
    }
}

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

def preprocess_image(image_path, target_size):
    """Preprocess image for prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, f"Failed to preprocess image: {str(e)}"

def predict_image(image_path, model, model_name):
    """Predict sign from image"""
    try:
        config = MODEL_CONFIG[model_name]
        labels = config["labels"]
        input_size = config["input_size"]
        
        # Preprocess image
        img_array, error = preprocess_image(image_path, input_size)
        if error:
            return {"text": error, "confidence": 0.0, "success": False}
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        # Get label
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
            "model_used": config["name"],
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
    model_name = sys.argv[2] if len(sys.argv) > 2 else "asl_alphabet"
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}", "text": "Image not found"}))
        sys.exit(1)
    
    # Load model
    model, error = load_model(model_name)
    if error:
        print(json.dumps({"error": error, "text": error}))
        sys.exit(1)
    
    # Predict
    result = predict_image(image_path, model, model_name)
    
    # Output JSON (use utf-8 encoding for Windows)
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
