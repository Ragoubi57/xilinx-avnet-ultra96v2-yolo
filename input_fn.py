import os
import cv2
import numpy as np

# --- CONFIG ---
CALIB_DIR = "calib_dataset"
INPUT_HEIGHT = 640
INPUT_WIDTH  = 640

# PASTE YOUR INPUT NODE NAME HERE (from Step 1 output)
# Example: input_node_name = "images" 
input_node_name = "images"  

def calib_input(iter):
    images = []
    files = os.listdir(CALIB_DIR)
    
    # Load batch of size 1 based on 'iter' index
    # We loop if we run out of images
    idx = iter % len(files)
    file_path = os.path.join(CALIB_DIR, files[idx])
    
    # Read Image
    img = cv2.imread(file_path)
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize (0-1)
    img = img.astype(np.float32) / 255.0
    
    # Add Batch Dimension (1, 640, 640, 3)
    img = np.expand_dims(img, 0)
    
    # Return dictionary mapping Node Name -> Data
    return {input_node_name: img}