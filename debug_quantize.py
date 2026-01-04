import tensorflow as tf
from tensorflow.python.framework import graph_util

def analyze_backbone():
    """Analyze what's actually in the backbone graph"""
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile("frozen_yolo_backbone.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
    
    print(f"Backbone graph: {len(graph_def.node)} nodes")
    
    ops = {}
    for n in graph_def.node:
        ops[n.op] = ops.get(n.op, 0) + 1
    
    print("\nAll ops:")
    for op, count in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
    
    return graph_def

def create_minimal_test_graph():
    """
    Create a tiny test graph to verify quantizer works at all.
    If this fails, it's a Vitis AI installation issue.
    """
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        # Simple: input -> conv -> output
        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")
        
        # Single conv layer
        w = tf.Variable(tf.random.normal([3, 3, 3, 16]), name="weights")
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name="conv1")
        
        # Add ReLU
        relu = tf.nn.relu(conv, name="relu1")
        
        # Output identity
        output = tf.identity(relu, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Freeze
        frozen = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ["output"]
        )
        
        with tf.io.gfile.GFile("test_minimal.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        
        print("Created test_minimal.pb")
        print("Ops:", [n.op for n in frozen.node])

def create_yolo_style_test():
    """
    Create a graph that mimics YOLO structure but is minimal.
    Tests if the issue is graph structure or specific ops.
    """
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        # YOLO-style input
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Conv block 1 (downsample)
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME', name="conv1")
        act1 = tf.nn.sigmoid(conv1, name="act1")  # SiLU approximation
        mul1 = tf.multiply(conv1, act1, name="silu1")
        
        # Conv block 2
        w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name="w2")
        conv2 = tf.nn.conv2d(mul1, w2, strides=[1, 2, 2, 1], padding='SAME', name="conv2")
        act2 = tf.nn.sigmoid(conv2, name="act2")
        mul2 = tf.multiply(conv2, act2, name="silu2")
        
        # Conv block 3
        w3 = tf.Variable(tf.random.normal([3, 3, 64, 128]), name="w3")
        conv3 = tf.nn.conv2d(mul2, w3, strides=[1, 2, 2, 1], padding='SAME', name="conv3")
        act3 = tf.nn.sigmoid(conv3, name="act3")
        mul3 = tf.multiply(conv3, act3, name="silu3")
        
        output = tf.identity(mul3, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        
        frozen = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ["output"]
        )
        
        with tf.io.gfile.GFile("test_yolo_style.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        
        print("\nCreated test_yolo_style.pb")
        ops = {}
        for n in frozen.node:
            ops[n.op] = ops.get(n.op, 0) + 1
        print("Ops:", ops)

def create_simple_calib_input():
    """Create a simple calibration input function"""
    code = '''
import numpy as np

def calib_input(iter):
    # For test_minimal.pb
    return {"input:0": np.random.rand(1, 224, 224, 3).astype(np.float32)}

def calib_input_yolo(iter):
    # For test_yolo_style.pb  
    return {"images:0": np.random.rand(1, 640, 640, 3).astype(np.float32)}
'''
    with open("test_calib.py", "w") as f:
        f.write(code)
    print("\nCreated test_calib.py")

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Analyze backbone")
    print("=" * 70)
    analyze_backbone()
    
    print("\n" + "=" * 70)
    print("STEP 2: Create minimal test graph")
    print("=" * 70)
    create_minimal_test_graph()
    
    print("\n" + "=" * 70)
    print("STEP 3: Create YOLO-style test graph")
    print("=" * 70)
    create_yolo_style_test()
    
    create_simple_calib_input()
    
    print("\n" + "=" * 70)
    print("NEXT: Test quantization with minimal graphs")
    print("=" * 70)
    print('''
# Test 1: Absolute minimal
vai_q_tensorflow quantize \\
  --input_frozen_graph test_minimal.pb \\
  --input_nodes input \\
  --input_shapes 1,224,224,3 \\
  --output_nodes output \\
  --input_fn test_calib.calib_input \\
  --output_dir quant_test1 \\
  --calib_iter 1

# Test 2: YOLO-style (with SiLU pattern)
vai_q_tensorflow quantize \\
  --input_frozen_graph test_yolo_style.pb \\
  --input_nodes images \\
  --input_shapes 1,640,640,3 \\
  --output_nodes output \\
  --input_fn test_calib.calib_input_yolo \\
  --output_dir quant_test2 \\
  --calib_iter 1
''')