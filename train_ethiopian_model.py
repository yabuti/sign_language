"""
Ethiopian Sign Language Model Training
Improved version with better data augmentation for robustness
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============ CONFIGURATION ============
# Change these paths to match your setup
TRAIN_DIR = "/content/drive/MyDrive/eth_frames/train"  # For Google Colab
VAL_DIR = "/content/drive/MyDrive/eth_frames/val"      # For Google Colab

# Or use local paths:
# TRAIN_DIR = r"C:\Users\PC\Desktop\sign-language-translator-deploy\eth_frames\train"
# VAL_DIR = r"C:\Users\PC\Desktop\sign-language-translator-deploy\eth_frames\val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = "eth_model_improved.keras"

# ============ DATA AUGMENTATION ============
# Strong augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,              # Rotate up to 30 degrees
    width_shift_range=0.3,          # Shift horizontally
    height_shift_range=0.3,         # Shift vertically
    shear_range=0.2,                # Shear transformation
    zoom_range=0.3,                 # Zoom in/out
    horizontal_flip=True,           # Flip horizontally
    brightness_range=[0.5, 1.5],    # Vary brightness (dark to bright)
    channel_shift_range=30,         # Color variation
    fill_mode='nearest'
)

# Validation data - only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# ============ LOAD DATA ============
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("\nLoading validation data...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Print class information
num_classes = train_generator.num_classes
print(f"\nNumber of classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ============ SAVE CLASS LABELS ============
# Important: Save this for prediction later!
import json
labels_path = "eth_labels.json"
with open(labels_path, 'w', encoding='utf-8') as f:
    # Reverse the dictionary: index -> label
    labels = {v: k for k, v in train_generator.class_indices.items()}
    json.dump(labels, f, ensure_ascii=False, indent=2)
print(f"\nClass labels saved to: {labels_path}")


# ============ BUILD MODEL ============
print("\nBuilding model...")

# Load pre-trained MobileNetV2 (without top layers)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# Build the full model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============ CALLBACKS ============
callbacks = [
    # Save best model
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Stop if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============ TRAIN PHASE 1: Frozen Base ============
print("\n" + "="*50)
print("PHASE 1: Training with frozen base model")
print("="*50)

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# ============ TRAIN PHASE 2: Fine-tuning ============
print("\n" + "="*50)
print("PHASE 2: Fine-tuning (unfreezing top layers)")
print("="*50)

# Unfreeze the top layers of base model
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=10,
    callbacks=callbacks
)

# ============ FINAL EVALUATION ============
print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

# Load best model
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Evaluate on validation set
val_loss, val_acc = best_model.evaluate(val_generator)
print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"✅ Labels saved to: {labels_path}")
print("\nDone! Copy these files to your project folder.")
