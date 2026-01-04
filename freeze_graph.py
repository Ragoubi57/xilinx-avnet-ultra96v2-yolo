import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os

# --- CONFIG ---
# Use the FIXED model folder you created earlier
INPUT_DIR = "yolo12_tf_fixed"
OUTPUT_FILE = "frozen_yolo.pb"

def main():
    print(f"Loading {INPUT_DIR}...")
    # Load the model with the signature we added
    loaded = tf.saved_model.load(INPUT_DIR)
    
    # Get the concrete function (the graph)
    infer = loaded.signatures['serving_default']
    
    # Convert variables to constants (Freeze)
    print("Freezing graph...")
    frozen_func = convert_variables_to_constants_v2(infer)
    frozen_func.graph.as_graph_def()

    # Save the file
    print(f"Saving to {OUTPUT_FILE}...")
    tf.io.write_graph(frozen_func.graph, ".", OUTPUT_FILE, as_text=False)
    
    # --- CRITICAL: PRINT NODE NAMES ---
    # We need these names for the next command
    print("\n" + "="*40)
    print("REQUIRED INFO FOR QUANTIZER")
    print("="*40)
    
    # Find input node name
    inputs = [t.name for t in frozen_func.inputs]
    print(f"INPUT NODE NAME:  {inputs[0].split(':')[0]}")
    
    # Find output node name
    outputs = [t.name for t in frozen_func.outputs]
    print(f"OUTPUT NODE NAME: {outputs[0].split(':')[0]}")
    print("="*40)

if __name__ == "__main__":
    main()