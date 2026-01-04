import tensorflow as tf
from tensorflow.python.framework import graph_util

INPUT_GRAPH = "frozen_yolo_clean.pb"
OUTPUT_GRAPH = "frozen_yolo_backbone.pb"

def find_pre_attention_outputs(graph_def):
    """
    Find the last Conv2D nodes BEFORE attention blocks (model.6, model.8).
    These are safe output points for DPU.
    """
    node_map = {n.name: n for n in graph_def.node}
    
    # Find all Conv2D nodes
    conv_nodes = [(n.name, n) for n in graph_def.node if n.op == 'Conv2D']
    
    print(f"Total Conv2D nodes: {len(conv_nodes)}")
    
    # Group by model layer
    # YOLO12 structure: model.0-5 (backbone), model.6 (attention), model.7, model.8 (attention), etc.
    pre_attention_convs = []
    attention_convs = []
    post_attention_convs = []
    
    for name, node in conv_nodes:
        # Check if this conv is BEFORE attention blocks
        if 'model.6' in name or 'model.8' in name:
            # These are in or after attention blocks
            if '/attn/' in name or 'matmul' in name.lower():
                attention_convs.append(name)
            else:
                post_attention_convs.append(name)
        elif any(f'model.{i}' in name for i in range(6)):
            # model.0 through model.5 - pure backbone
            pre_attention_convs.append(name)
        elif 'model.9' in name or 'model.10' in name:
            # Neck - after first attention block but before detection head
            post_attention_convs.append(name)
        else:
            post_attention_convs.append(name)
    
    print(f"\nPre-attention convs (backbone): {len(pre_attention_convs)}")
    print(f"Attention-related convs: {len(attention_convs)}")
    print(f"Post-attention convs: {len(post_attention_convs)}")
    
    return pre_attention_convs, post_attention_convs

def find_feature_pyramid_outputs(graph_def):
    """
    Find the FPN/PAN output nodes - these feed into the detection head.
    For YOLO, these are typically the last convs at each scale level.
    """
    conv_nodes = [n.name for n in graph_def.node if n.op == 'Conv2D']
    
    # Find convs that might be scale outputs (look for patterns)
    # In YOLO, FPN outputs are usually around model.4, model.6 outputs, model.9, etc.
    
    # Let's find the last few convs at different "levels"
    # Group by what looks like layer indices
    
    candidates = []
    
    # Look for specific patterns that indicate feature map outputs
    for name in conv_nodes:
        # Skip attention-related convs
        if '/attn/' in name:
            continue
        # These patterns often indicate FPN outputs in YOLO
        if any(p in name for p in ['model.4/', 'model.5/', 'model.3/']):
            candidates.append(name)
    
    print(f"\nCandidate FPN output convs: {len(candidates)}")
    for c in candidates[-10:]:
        print(f"  {c}")
    
    return candidates

def extract_backbone():
    print(f"Loading {INPUT_GRAPH}...")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(INPUT_GRAPH, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    print(f"Original graph: {len(graph_def.node)} nodes")
    
    # Analyze structure
    pre_attn, post_attn = find_pre_attention_outputs(graph_def)
    
    # Find all conv nodes and their layer info
    conv_nodes = []
    for n in graph_def.node:
        if n.op == 'Conv2D':
            conv_nodes.append(n.name)
    
    # Let's find the LAST convolutions before any attention-related ops
    # by looking at node order and patterns
    
    print("\n" + "=" * 70)
    print("Looking for safe backbone outputs...")
    print("=" * 70)
    
    # Find convs that DON'T have attention ops in their downstream path
    # For simplicity, let's find the last convs in model.5 (before model.6 attention)
    
    model5_convs = [n for n in conv_nodes if 'model.5' in n or 'model_5' in n]
    model4_convs = [n for n in conv_nodes if 'model.4' in n or 'model_4' in n]
    model3_convs = [n for n in conv_nodes if 'model.3' in n or 'model_3' in n]
    
    print(f"\nmodel.3 convs: {len(model3_convs)}")
    print(f"model.4 convs: {len(model4_convs)}")
    print(f"model.5 convs: {len(model5_convs)}")
    
    # Use the last conv from each backbone stage as outputs
    # These are the feature pyramid inputs before attention
    output_candidates = []
    
    if model3_convs:
        output_candidates.append(model3_convs[-1])
        print(f"\nmodel.3 output: {model3_convs[-1]}")
    if model4_convs:
        output_candidates.append(model4_convs[-1])
        print(f"model.4 output: {model4_convs[-1]}")
    if model5_convs:
        output_candidates.append(model5_convs[-1])
        print(f"model.5 output: {model5_convs[-1]}")
    
    if not output_candidates:
        # Fallback: use last N convs that don't have attention patterns
        safe_convs = [n for n in conv_nodes if '/attn/' not in n and 'matmul' not in n.lower()]
        # Find convs before model.6
        for n in safe_convs:
            if not any(f'model.{i}' in n for i in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]):
                output_candidates.append(n)
        output_candidates = output_candidates[-3:]  # Last 3
        print(f"\nFallback outputs: {output_candidates}")
    
    if not output_candidates:
        print("ERROR: Could not find safe output nodes!")
        return
    
    print("\n" + "=" * 70)
    print(f"Extracting subgraph with {len(output_candidates)} outputs...")
    print("=" * 70)
    
    try:
        sub_graph_def = graph_util.extract_sub_graph(graph_def, output_candidates)
        print(f"Extracted graph: {len(sub_graph_def.node)} nodes")
        
        # Verify no problematic ops
        ops = {}
        for n in sub_graph_def.node:
            ops[n.op] = ops.get(n.op, 0) + 1
        
        print("\nOps in extracted graph:")
        for op, count in sorted(ops.items(), key=lambda x: -x[1]):
            print(f"  {op}: {count}")
        
        # Check for problems
        problems = [op for op in ops if op in {'BatchMatMulV2', 'Softmax', 'MatMul'}]
        if problems:
            print(f"\n⚠️ WARNING: Still has problematic ops: {problems}")
        else:
            print(f"\n✓ No problematic ops found!")
        
        # Save
        print(f"\nSaving to {OUTPUT_GRAPH}...")
        with tf.io.gfile.GFile(OUTPUT_GRAPH, "wb") as f:
            f.write(sub_graph_def.SerializeToString())
        
        print("\n" + "=" * 70)
        print("NEXT STEP:")
        print("=" * 70)
        outputs_str = ",".join(output_candidates)
        print(f'''
vai_q_tensorflow quantize \\
  --input_frozen_graph {OUTPUT_GRAPH} \\
  --input_nodes images \\
  --input_shapes 1,640,640,3 \\
  --output_nodes {outputs_str} \\
  --input_fn input_fn.calib_input \\
  --output_dir quant_output \\
  --calib_iter 10
''')
        
    except Exception as e:
        print(f"ERROR extracting subgraph: {e}")

if __name__ == "__main__":
    extract_backbone()