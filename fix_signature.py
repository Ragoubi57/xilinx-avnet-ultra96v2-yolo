import tensorflow as tf
import os

# --- CONFIG ---
INPUT_DIR  = "yolo12_tf_model"   # The folder with the missing signature
OUTPUT_DIR = "yolo12_tf_fixed"   # The new folder we will create

def main():
    print(f"Loading model from {INPUT_DIR}...")
    try:
        # Load the raw SavedModel
        obj = tf.saved_model.load(INPUT_DIR)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        return

    print("Model loaded. Creating new signature...")

    # Define the input shape (YOLO Standard: Batch=1, Height=640, Width=640, Channels=3)
    # onnx2tf converts inputs to NHWC format.
    input_shape = (1, 640, 640, 3)
    
    # Create a concrete function that wraps the model's call method
    # We give the input a specific name 'images'
    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='images')])
    def serving_fn(images):
        return obj(images)

    print(f"Saving new model to {OUTPUT_DIR}...")
    
    # Save it back with the 'serving_default' signature
    tf.saved_model.save(obj, OUTPUT_DIR, signatures={'serving_default': serving_fn})
    
    print("SUCCESS!")
    print(f"New model saved in: {OUTPUT_DIR}")
    print("Update your quantization script to point to this new folder.")

if __name__ == "__main__":
    main()