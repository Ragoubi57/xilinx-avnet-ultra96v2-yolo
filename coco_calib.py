import os
import shutil
from onnx2tf import convert

# --- CONFIGURATION ---
# The name of your ONNX file
ONNX_FILE = "yolo12n_op12_static_1_640.onnx"
# The output folder name
OUTPUT_FOLDER = "yolo12_tf_model"

def main():
    # 1. Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, ONNX_FILE)
    output_path = os.path.join(current_dir, OUTPUT_FOLDER)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # 2. Check if file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {ONNX_FILE}")
        print("Make sure this script is in the same folder as the .onnx file!")
        return

    # 3. Clean previous output if exists
    if os.path.exists(output_path):
        print("Removing old output folder...")
        shutil.rmtree(output_path)

    # 4. Run Conversion
    print("\nStarting conversion (this may take a minute)...")
    try:
        # Convert ONNX to TF SavedModel
        # output_signature_defs=True helps Vitis identify inputs/outputs
        # ... inside the try block ...
        convert(
            input_onnx_file_path=input_path,
            output_folder_path=output_path,
            output_signature_defs=True,
            disable_group_convolution=True  # <--- ADD THIS LINE
        )
        print("\nSUCCESS!")
        print(f"TensorFlow model saved to: {output_path}")
        print("You can now switch to Docker and run the quantization script.")
        
    except Exception as e:
        print(f"\nFAILURE: Conversion failed.\nError: {e}")

if __name__ == "__main__":
    main()