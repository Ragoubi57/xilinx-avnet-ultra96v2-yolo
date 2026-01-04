import tensorflow as tf

GRAPH_PB_PATH = 'frozen_yolo.pb'

def main():
    print(f"Inspecting {GRAPH_PB_PATH}...")
    with tf.io.gfile.GFile(GRAPH_PB_PATH, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    print("\n--- POSSIBLE INPUT NODES ---")
    # Inputs are usually 'Placeholder' operations
    for node in graph_def.node:
        if node.op == 'Placeholder':
            print(f"Name: {node.name}  (Shape: {node.attr['shape']})")

    print("\n--- POSSIBLE OUTPUT NODES ---")
    # The output is usually the very last node in the file
    last_node = graph_def.node[-1]
    print(f"Last Node: {last_node.name} (Op: {last_node.op})")
    
    # Sometimes it's the one before the last if the last is NoOp
    if len(graph_def.node) > 1:
        prev = graph_def.node[-2]
        print(f"Prev Node: {prev.name} (Op: {prev.op})")

if __name__ == "__main__":
    main()