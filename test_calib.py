import numpy as np

def calib_input(iter):
    # For test_minimal.pb - key WITHOUT :0
    return {"input": np.random.rand(1, 224, 224, 3).astype(np.float32)}

def calib_input_yolo(iter):
    # For test_yolo_style.pb - key WITHOUT :0
    return {"images": np.random.rand(1, 640, 640, 3).astype(np.float32)}