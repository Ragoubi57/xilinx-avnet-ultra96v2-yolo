import tensorflow as tf

INPUT_GRAPH = "frozen_yolo_stripped.pb"
OUTPUT_GRAPH = "frozen_yolo_dpu_only.pb"

# Operators that crash Vitis AI 2.5 quantizer
CRASH_OPS = {'BatchMatMulV2', 'BatchMatMul', 'Softmax'}

def remove_crashing_ops():
    print(f"Loading {INPUT_GRAPH}...")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(INPUT_GRAPH, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    print(f"Original graph has {len(graph_def.node)} nodes")
    
    # Count problematic ops
    crash_nodes = [(n.name, n.op) for n in graph_def.node if n.op in CRASH_OPS]
    print(f"Found {len(crash_nodes)} nodes that crash quantizer:")
    for name, op in crash_nodes[:5]:
        print(f"  [{op}] {name}")
    if len(crash_nodes) > 5:
        print(f"  ... and {len(crash_nodes) - 5} more")
    
    # Build dependency map: which nodes feed into crash ops
    node_map = {n.name: n for n in graph_def.node}
    crash_node_names = {n.name for n in graph_def.node if n.op in CRASH_OPS}
    
    # Find all nodes that are consumed ONLY by crash ops (we can remove these too)
    # But for safety, we'll just replace crash ops with Identity
    
    print("\n" + "=" * 60)
    print("Replacing crash-inducing ops with Identity...")
    print("=" * 60)
    
    new_graph_def = tf.compat.v1.GraphDef()
    converted = 0
    
    for node in graph_def.node:
        new_node = new_graph_def.node.add()
        
        if node.op in CRASH_OPS:
            # Replace with Identity
            new_node.name = node.name
            new_node.op = "Identity"
            
            # Take only the first input
            if node.input:
                new_node.input.append(node.input[0])
            
            # Set dtype
            new_node.attr['T'].type = tf.float32.as_datatype_enum
            
            converted += 1
        else:
            # Copy as-is
            new_node.CopyFrom(node)
    
    print(f"Converted {converted} ops to Identity")
    
    # Verify
    remaining = [n.op for n in new_graph_def.node if n.op in CRASH_OPS]
    if remaining:
        print(f"ERROR: Still have crash ops: {remaining}")
        return False
    
    print("✓ All crash-inducing ops removed!")
    
    # Save
    print(f"\nSaving to {OUTPUT_GRAPH}...")
    with tf.io.gfile.GFile(OUTPUT_GRAPH, "wb") as f:
        f.write(new_graph_def.SerializeToString())
    
    # Print ops summary
    op_counts = {}
    for n in new_graph_def.node:
        op_counts[n.op] = op_counts.get(n.op, 0) + 1
    
    print("\n" + "=" * 60)
    print("NEW GRAPH OPS:")
    print("=" * 60)
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {op}: {count}")
    
    print("\n" + "=" * 60)
    print("NEXT: Run quantization with new graph")
    print("=" * 60)
    print(f'''
vai_q_tensorflow quantize \\
  --input_frozen_graph {OUTPUT_GRAPH} \\
  --input_nodes images \\
  --input_shapes 1,640,640,3 \\
  --output_nodes PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_871/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_872/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_873/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_874/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_875/convolution \\
  --input_fn input_fn.calib_input \\
  --output_dir quant_output \\
  --calib_iter 20
''')
    
    return True

if __name__ == "__main__":
    remove_crashing_ops()