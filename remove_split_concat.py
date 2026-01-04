import tensorflow as tf

INPUT_GRAPH = "frozen_yolo_stripped.pb"
OUTPUT_GRAPH = "frozen_yolo_no_split.pb"

# Ops that crash Vitis AI 2.5
CRASH_OPS = {
    'BatchMatMulV2', 'BatchMatMul', 'Softmax',  # Attention ops
    'Split', 'SplitV',  # Split ops - CRASH!
    'ConcatV2',  # Concat ops - CRASH!
}

def remove_crash_ops():
    print(f"Loading {INPUT_GRAPH}...")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(INPUT_GRAPH, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    print(f"Original: {len(graph_def.node)} nodes")
    
    # Count crash ops
    crash_counts = {}
    for n in graph_def.node:
        if n.op in CRASH_OPS:
            crash_counts[n.op] = crash_counts.get(n.op, 0) + 1
    
    print(f"\nCrash-inducing ops found:")
    for op, count in crash_counts.items():
        print(f"  {op}: {count}")
    
    # Build node map
    node_map = {n.name: n for n in graph_def.node}
    
    # Replace crash ops with Identity
    new_graph_def = tf.compat.v1.GraphDef()
    converted = 0
    
    for node in graph_def.node:
        new_node = new_graph_def.node.add()
        
        if node.op in CRASH_OPS:
            new_node.name = node.name
            new_node.op = "Identity"
            
            # For Split/SplitV/ConcatV2, take the first data input (skip axis input)
            if node.op in ['Split', 'SplitV']:
                # Split: input 0 is axis (scalar), input 1 is data
                # SplitV: input 0 is data, input 1 is size_splits, input 2 is axis
                if node.op == 'Split' and len(node.input) >= 2:
                    new_node.input.append(node.input[1])  # data input
                elif node.op == 'SplitV' and len(node.input) >= 1:
                    new_node.input.append(node.input[0])  # data input
                else:
                    new_node.input.append(node.input[0])
            elif node.op == 'ConcatV2':
                # ConcatV2: last input is axis, others are tensors to concat
                # Just pass through first tensor
                if node.input:
                    new_node.input.append(node.input[0])
            else:
                # BatchMatMulV2, Softmax, etc.
                if node.input:
                    new_node.input.append(node.input[0])
            
            new_node.attr['T'].type = tf.float32.as_datatype_enum
            converted += 1
        else:
            new_node.CopyFrom(node)
    
    print(f"\nConverted {converted} ops to Identity")
    
    # Verify
    remaining = {}
    for n in new_graph_def.node:
        if n.op in CRASH_OPS:
            remaining[n.op] = remaining.get(n.op, 0) + 1
    
    if remaining:
        print(f"WARNING: Still have crash ops: {remaining}")
    else:
        print("✓ All crash ops removed!")
    
    # Count new ops
    ops = {}
    for n in new_graph_def.node:
        ops[n.op] = ops.get(n.op, 0) + 1
    
    print(f"\nNew graph ops:")
    for op, count in sorted(ops.items(), key=lambda x: -x[1])[:10]:
        print(f"  {op}: {count}")
    
    # Save
    with tf.io.gfile.GFile(OUTPUT_GRAPH, "wb") as f:
        f.write(new_graph_def.SerializeToString())
    
    print(f"\nSaved to {OUTPUT_GRAPH}")
    print("\n" + "=" * 70)
    print("NEXT: Try quantization")
    print("=" * 70)
    print(f'''
vai_q_tensorflow quantize \\
  --input_frozen_graph {OUTPUT_GRAPH} \\
  --input_nodes images \\
  --input_shapes 1,640,640,3 \\
  --output_nodes PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_871/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_872/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_873/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_874/convolution,PartitionedCall/PartitionedCall/model_41/tf.nn.convolution_875/convolution \\
  --input_fn input_fn.calib_input \\
  --output_dir quant_output \\
  --calib_iter 5
''')

if __name__ == "__main__":
    remove_crash_ops()