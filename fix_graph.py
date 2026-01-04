import tensorflow as tf

# --- CONFIG ---
INPUT_FILE = "frozen_yolo.pb"
OUTPUT_FILE = "frozen_yolo_clean.pb"

def main():
    print(f"Loading {INPUT_FILE}...")
    graph_def = tf.compat.v1.GraphDef()
    
    try:
        with open(INPUT_FILE, "rb") as f:
            graph_def.ParseFromString(f.read())
    except Exception as e:
        print(f"Error parsing graph: {e}")
        return

    print("Scanning nodes for TF2-only attributes...")
    count = 0
    
    for node in graph_def.node:
        # 1. Fix MatMuls (grad_x, grad_y)
        if node.op == 'BatchMatMulV2':
            if 'grad_x' in node.attr:
                del node.attr['grad_x']
                count += 1
            if 'grad_y' in node.attr:
                del node.attr['grad_y']
                count += 1
                
        # 2. Fix Convolutions (explicit_paddings)
        # This affects Conv2D, DepthwiseConv2dNative, MaxPool, etc.
        if 'explicit_paddings' in node.attr:
            # print(f" - Fixing node: {node.name} (Removed explicit_paddings)")
            del node.attr['explicit_paddings']
            count += 1

        # 3. Fix BatchNorm (U)
        if node.op == 'FusedBatchNormV3':
            if 'U' in node.attr:
                del node.attr['U']
                count += 1

    print(f"\nFixed {count} attributes.")
    
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        f.write(graph_def.SerializeToString())
    print("Done.")

if __name__ == "__main__":
    main()