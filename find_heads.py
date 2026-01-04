import tensorflow as tf

GRAPH_PATH = "frozen_yolo_clean.pb"

def main():
    print(f"Scanning {GRAPH_PATH}...")
    with tf.io.gfile.GFile(GRAPH_PATH, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Find all Conv2D nodes
    conv_nodes = [n.name for n in graph_def.node if n.op == 'Conv2D']
    
    print(f"\nFound {len(conv_nodes)} Conv2D layers.")
    print("Here are the last 5 Conv2D nodes (likely your output heads):")
    print("-" * 50)
    
    # Print the last 5
    for name in conv_nodes[-5:]:
        print(name)
    print("-" * 50)
    print("Use these names (comma separated) as your --output_nodes")

if __name__ == "__main__":
    main()