import tensorflow as tf

def analyze_graph(graph_path):
    print(f"Loading {graph_path}...")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    print(f"Total nodes: {len(graph_def.node)}")
    
    # Known problematic ops for Vitis AI 2.5
    KNOWN_UNSUPPORTED = {
        'BatchMatMulV2', 'BatchMatMul', 'MatMul',  # Matrix ops
        'Softmax', 'LogSoftmax',                    # Softmax variants
        'GatherV2', 'Gather', 'ScatterNd',          # Gather/Scatter
        'Where', 'Select', 'SelectV2',              # Conditional
        'Range', 'Fill',                            # Dynamic shape
        'TensorArrayV3', 'TensorArrayReadV3',       # Tensor arrays
        'ExpandDims', 'Squeeze',                    # Shape manipulation (sometimes)
        'StridedSlice',                             # Complex slicing
        'Pack', 'Unpack',                           # Pack/Unpack
        'Tile',                                     # Tile op
        'Cast',                                     # Type casting
    }
    
    # Potentially problematic ops
    SUSPICIOUS = {
        'Reshape', 'Transpose', 'Split', 'SplitV', 'ConcatV2',
        'ResizeNearestNeighbor', 'ResizeBilinear'
    }
    
    # Collect stats
    op_counts = {}
    unsupported_nodes = []
    suspicious_nodes = []
    
    for node in graph_def.node:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
        
        if node.op in KNOWN_UNSUPPORTED:
            unsupported_nodes.append((node.name, node.op, len(node.input)))
        elif node.op in SUSPICIOUS:
            suspicious_nodes.append((node.name, node.op, len(node.input)))
    
    print("\n" + "=" * 70)
    print("ALL OPERATORS IN GRAPH:")
    print("=" * 70)
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        status = ""
        if op in KNOWN_UNSUPPORTED:
            status = " ❌ UNSUPPORTED"
        elif op in SUSPICIOUS:
            status = " ⚠️  SUSPICIOUS"
        print(f"  {op}: {count}{status}")
    
    print("\n" + "=" * 70)
    print(f"UNSUPPORTED NODES ({len(unsupported_nodes)}):")
    print("=" * 70)
    for name, op, num_inputs in unsupported_nodes:
        print(f"  [{op}] {name} (inputs: {num_inputs})")
    
    print("\n" + "=" * 70)
    print(f"SUSPICIOUS NODES ({len(suspicious_nodes)}):")
    print("=" * 70)
    # Group by op type
    by_op = {}
    for name, op, num_inputs in suspicious_nodes:
        if op not in by_op:
            by_op[op] = []
        by_op[op].append(name)
    
    for op, names in by_op.items():
        print(f"\n  {op} ({len(names)} nodes):")
        for name in names[:3]:
            print(f"    - {name}")
        if len(names) > 3:
            print(f"    ... and {len(names) - 3} more")
    
    # Check for dynamic shapes (common crash cause)
    print("\n" + "=" * 70)
    print("CHECKING FOR DYNAMIC SHAPE ISSUES:")
    print("=" * 70)
    
    dynamic_shape_ops = []
    for node in graph_def.node:
        # Check for shape-related ops that might cause issues
        if node.op in ['Shape', 'ShapeN', 'Size', 'Rank']:
            dynamic_shape_ops.append((node.name, node.op))
        # Check Reshape with non-constant shape
        if node.op == 'Reshape':
            # Second input is the shape - check if it's a Const
            if len(node.input) >= 2:
                shape_input = node.input[1].split(':')[0]
                shape_node = next((n for n in graph_def.node if n.name == shape_input), None)
                if shape_node and shape_node.op != 'Const':
                    dynamic_shape_ops.append((node.name, f"Reshape with dynamic shape from {shape_node.op}"))
    
    if dynamic_shape_ops:
        print("  Found potentially problematic dynamic shape ops:")
        for name, op in dynamic_shape_ops[:10]:
            print(f"    [{op}] {name}")
    else:
        print("  No obvious dynamic shape issues found")
    
    return op_counts, unsupported_nodes, suspicious_nodes

if __name__ == "__main__":
    print("=" * 70)
    print("ANALYZING: frozen_yolo_dpu_only.pb")
    print("=" * 70)
    analyze_graph("frozen_yolo_dpu_only.pb")
    
    print("\n\n")
    print("=" * 70)
    print("COMPARING WITH: frozen_yolo_stripped.pb")
    print("=" * 70)
    analyze_graph("frozen_yolo_stripped.pb")