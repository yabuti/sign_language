#!/usr/bin/env python3
"""
Test script to verify models can be loaded
"""

import os
from tensorflow import keras

print("🔍 Testing Model Loading...\n")

models = {
    "CNN-LSTM Model": "asl_best_cnn_lstm.keras",
    "MobileNet Model": "eth_model_mobilenet_best.keras",
    "Legacy LSTM": "asl_model_lstm.h5"
}

for name, path in models.items():
    print(f"Testing {name}...")
    if os.path.exists(path):
        try:
            model = keras.models.load_model(path)
            print(f"  ✅ Loaded successfully")
            print(f"  📊 Input shape: {model.input_shape}")
            print(f"  📊 Output shape: {model.output_shape}")
            print(f"  📊 Total params: {model.count_params():,}")
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")
    else:
        print(f"  ⚠️  File not found: {path}")
    print()

print("✅ Model testing complete!")
