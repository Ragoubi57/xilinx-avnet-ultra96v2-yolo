import os
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# --- CONFIG ---
# Points to the FIXED model
INPUT_MODEL_DIR = "./yolo12_tf_fixed" 
CALIB_DIR       = "./calib_dataset"
OUTPUT_DIR      = "./quant_output"
INPUT_SHAPE     = (640, 640)

def load_data():
    # Load 30 images for calibration
    files = [f for f in os.listdir(CALIB_DIR) if f.endswith(('.jpg','.png'))][:30]
    print(f"Calibrating with {len(files)} images...")
    
    for f in files:
        path = os.path.join(CALIB_DIR, f)
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, INPUT_SHAPE)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)
        yield [img]

def main():
    # 1. Load
    print("Loading fixed model...")
    model = tf.keras.models.load_model(INPUT_MODEL_DIR)
    
    # 2. Quantize
    print("Starting Vitis Quantizer...")
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(
        calib_dataset=load_data,
        calib_batch_size=1,
        calib_steps=30
    )
    
    # 3. Save
    print(f"Saving to {OUTPUT_DIR}...")
    quantized_model.save(os.path.join(OUTPUT_DIR, "quantized.h5"))
    print("Quantization Complete.")

if __name__ == "__main__":
    main()