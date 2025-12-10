#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Model Server - Keeps models loaded in memory for fast predictions
Run with: python model_server.py
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow import keras
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ===== Global Model Cache =====
MODELS = {}
MODELS_LOADED = False

# ===== Model Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "asl_alphabet": os.path.join(BASE_DIR, "..", "frontend", "sign_language_model.h5"),
    "ethiopian": os.path.join(BASE_DIR, "..", "eth_model_mobilenet_best.keras"),
}

# ===== Labels =====
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

ETH_LABELS = [
    'ሂድ', 'ህመም', 'መንደር', 'ምግብ', 'ሰላም', 'ቀለም', 'አመሰግናለሁ', 'አቁም', 'አዎን', 'እባክህ',
    'እንደገና', 'እገዛ', 'እግር', 'ውሃ', 'ይቅርታ', 'ድምፅ', 'ድንጋይ', 'ግራ', 'ጥሩ', 'ጨምር'
]

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

def load_all_models():
    """Load all models at startup - only once"""
    global MODELS, MODELS_LOADED
    
    if MODELS_LOADED:
        return
    
    print("[INFO] Loading models into memory...")
    
    for model_name, model_path in MODEL_PATHS.items():
        try:
            print(f"  Checking path: {model_path}")
            if os.path.exists(model_path):
                print(f"  Loading {model_name}...")
                MODELS[model_name] = keras.models.load_model(model_path)
                # Warm up the model with a dummy prediction
                config = MODEL_CONFIG[model_name]
                dummy_input = np.zeros((1, *config["input_size"], 3), dtype=np.float32)
                MODELS[model_name].predict(dummy_input, verbose=0)
                print(f"  [OK] {model_name} loaded and warmed up!")
            else:
                print(f"  [WARN] Model not found: {model_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_name}: {e}")
    
    MODELS_LOADED = True
    print(f"[OK] All models loaded! ({len(MODELS)} models ready)")


def preprocess_image(image_bytes, target_size):
    """Preprocess image for prediction"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"success": False, "text": "No image provided"})
        
        image_file = request.files['image']
        model_name = request.form.get('model', 'asl_alphabet')
        
        # Check if model is loaded
        if model_name not in MODELS:
            return jsonify({
                "success": False, 
                "text": f"Model '{model_name}' not available"
            })
        
        model = MODELS[model_name]
        config = MODEL_CONFIG[model_name]
        labels = config["labels"]
        
        # Preprocess image
        image_bytes = image_file.read()
        img_array = preprocess_image(image_bytes, config["input_size"])
        
        # Predict (fast - model already loaded!)
        predictions = model.predict(img_array, verbose=0)
        
        # Get results
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_label = labels[predicted_index] if predicted_index < len(labels) else f"Sign_{predicted_index}"
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            label = labels[idx] if idx < len(labels) else f"Sign_{idx}"
            top_3.append({
                "label": label,
                "confidence": round(float(predictions[0][idx]) * 100, 2)
            })
        
        return jsonify({
            "success": True,
            "text": predicted_label,
            "confidence": round(confidence * 100, 2),
            "top_3": top_3,
            "model_used": config["name"]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "text": f"Error: {str(e)}"
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "models_count": len(MODELS)
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "available_models": [
            {"id": k, "name": v["name"], "loaded": k in MODELS}
            for k, v in MODEL_CONFIG.items()
        ]
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Sign Language Model Server")
    print("=" * 50)
    
    # Load models at startup
    load_all_models()
    
    print("\nServer starting on http://localhost:5000")
    print("   - POST /predict - Image prediction")
    print("   - GET /health - Health check")
    print("   - GET /models - List models")
    print("=" * 50)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
