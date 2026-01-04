import tensorflow as tf
from tensorflow.python.framework import graph_util

# --- CONFIG ---
INPUT_GRAPH = "frozen_yolo_clean.pb"
OUTPUT_GRAPH = "frozen_yolo_stripped.pb"

# PASTE YOUR EXACT OUTPUT NODES HERE (Comma separated, no spaces)
OUTPUT_NODES = [
    "PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_871/convolution",
    "PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_872/convolution",
    "PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_873/convolution",
    "PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_874/convolution",
    "PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_875/convolution"
]

def main():
    print(f"Loading {INPUT_GRAPH}...")
    
    # Load the graph definition
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(INPUT_GRAPH, "rb") as f:
        graph_def.ParseFromString(f.read())

    print(f"Original graph has {len(graph_def.node)} nodes.")
    print("Extracting subgraph (Stripping dangerous post-processing)...")

    # This function walks backwards from your outputs and deletes everything else
    try:
        sub_graph_def = graph_util.extract_sub_graph(
            graph_def, 
            OUTPUT_NODES
        )
    except Exception as e:
        print(f"\nERROR: Could not extract subgraph. Check your node names!")
        print(f"Details: {e}")
        return

    print(f"New graph has {len(sub_graph_def.node)} nodes.")
    
    print(f"Saving to {OUTPUT_GRAPH}...")
    with tf.io.gfile.GFile(OUTPUT_GRAPH, "wb") as f:
        f.write(sub_graph_def.SerializeToString())
    print("Done. You can now quantize this file safely.")

if __name__ == "__main__":
    main()