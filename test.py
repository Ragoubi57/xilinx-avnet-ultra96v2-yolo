import tensorflow as tf

INPUT_GRAPH = "frozen_yolo_stripped.pb"

def analyze_graph():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(INPUT_GRAPH, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    # Count operators
    op_counts = {}
    unsupported_nodes = []
    
    UNSUPPORTED_OPS = {'BatchMatMulV2', 'BatchMatMul', 'MatMul', 'Softmax', 'GatherV2', 'ScatterNd'}
    
    for node in graph_def.node:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
        if node.op in UNSUPPORTED_OPS:
            unsupported_nodes.append((node.name, node.op, list(node.input)))
    
    print("=" * 60)
    print("OPERATOR COUNTS (sorted by frequency)")
    print("=" * 60)
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        marker = " ❌ UNSUPPORTED" if op in UNSUPPORTED_OPS else ""
        print(f"  {op}: {count}{marker}")
    
    print("\n" + "=" * 60)
    print(f"UNSUPPORTED NODES DETAIL ({len(unsupported_nodes)} total)")
    print("=" * 60)
    for name, op, inputs in unsupported_nodes[:20]:  # First 20
        print(f"\n  Node: {name}")
        print(f"  Op:   {op}")
        print(f"  Inputs: {inputs[:3]}...")  # First 3 inputs
    
    return unsupported_nodes

if __name__ == "__main__":
    nodes = analyze_graph()
    print(f"\n\n⚠️  Found {len(nodes)} unsupported operations.")
    print("These MUST be removed or replaced for Vitis AI 2.5 quantization.")