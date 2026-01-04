import tensorflow as tf
import os

MODEL_PATH = "yolo12_tf_model"

print(f"Loading {MODEL_PATH}...")
try:
    model = tf.saved_model.load(MODEL_PATH)
    print("\n--- AVAILABLE SIGNATURES ---")
    print(list(model.signatures.keys()))
    
    # Also print the trackable attributes to see if we can call it directly
    print("\n--- MODEL ATTRIBUTES ---")
    print(dir(model))
except Exception as e:
    print(f"Error: {e}")