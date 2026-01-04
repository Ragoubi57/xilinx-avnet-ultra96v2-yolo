import tensorflow as tf
import cv2
import numpy as np
import os

# --- CONFIG ---
MODEL_PATH = "yolo12_tf_fixed"
IMG_DIR    = "calib_dataset"
INPUT_SIZE = 640

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.saved_model.load(MODEL_PATH)
        inference_func = model.signatures["serving_default"]
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get input tensor name (usually 'images' or 'input')
    input_tensor_name = list(inference_func.structured_input_signature[1].keys())[0]
    print(f"Model expects input named: '{input_tensor_name}'")

    # Load one image
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not img_files:
        print("No images found in calib_dataset!")
        return
    
    img_path = os.path.join(IMG_DIR, img_files[0])
    print(f"Testing with image: {img_path}")

    # Preprocess (Standard YOLO: Resize -> RGB -> Norm -> NHWC)
    # Note: onnx2tf converts models to NHWC (height, width, channel)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0) # Add batch dim -> (1, 640, 640, 3)

    # Run Inference
    print("Running inference...")
    input_tensor = tf.convert_to_tensor(img)
    output = inference_func(**{input_tensor_name: input_tensor})

    # Print Result Keys and Shapes
    print("\n--- SUCCESS ---")
    for key, value in output.items():
        print(f"Output '{key}': Shape {value.shape}")
        # Expected YOLO output shape is usually (1, 84, 8400) or similar
        
if __name__ == "__main__":
    main()