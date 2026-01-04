import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np

def create_test_with_transpose():
    """Test if Transpose causes the crash"""
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Conv
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')
        
        # Transpose (NHWC -> NCHW -> NHWC)
        trans1 = tf.transpose(conv1, [0, 3, 1, 2])  # NHWC -> NCHW
        trans2 = tf.transpose(trans1, [0, 2, 3, 1])  # NCHW -> NHWC
        
        # Another conv
        w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name="w2")
        conv2 = tf.nn.conv2d(trans2, w2, strides=[1, 2, 2, 1], padding='SAME')
        
        output = tf.identity(conv2, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        frozen = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        
        with tf.io.gfile.GFile("test_transpose.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        print("Created test_transpose.pb")

def create_test_with_reshape():
    """Test if Reshape causes the crash"""
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Conv
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')  # [1,320,320,32]
        
        # Reshape
        reshaped = tf.reshape(conv1, [1, 320, 320, 32])  # Same shape, explicit
        
        # Another conv
        w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name="w2")
        conv2 = tf.nn.conv2d(reshaped, w2, strides=[1, 2, 2, 1], padding='SAME')
        
        output = tf.identity(conv2, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        frozen = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        
        with tf.io.gfile.GFile("test_reshape.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        print("Created test_reshape.pb")

def create_test_with_split_concat():
    """Test if Split/Concat causes the crash"""
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Conv
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')  # [1,320,320,32]
        
        # Split along channel axis
        split1, split2 = tf.split(conv1, 2, axis=3)  # Two [1,320,320,16] tensors
        
        # Process each
        w2 = tf.Variable(tf.random.normal([3, 3, 16, 16]), name="w2")
        conv2 = tf.nn.conv2d(split1, w2, strides=[1, 1, 1, 1], padding='SAME')
        
        # Concat back
        concat = tf.concat([conv2, split2], axis=3)  # [1,320,320,32]
        
        # Final conv
        w3 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name="w3")
        conv3 = tf.nn.conv2d(concat, w3, strides=[1, 2, 2, 1], padding='SAME')
        
        output = tf.identity(conv3, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        frozen = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        
        with tf.io.gfile.GFile("test_split_concat.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        print("Created test_split_concat.pb")

def create_test_with_identity_chain():
    """Test if many Identity ops cause issues"""
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Conv
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')
        
        # Chain of identities (YOLO12 has 2055 Identity nodes!)
        current = conv1
        for i in range(100):
            current = tf.identity(current, name=f"id_{i}")
        
        # Final conv
        w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name="w2")
        conv2 = tf.nn.conv2d(current, w2, strides=[1, 2, 2, 1], padding='SAME')
        
        output = tf.identity(conv2, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        frozen = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        
        with tf.io.gfile.GFile("test_identity_chain.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        print("Created test_identity_chain.pb")

def create_test_complex():
    """Combine multiple suspicious ops"""
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Conv block with SiLU
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')
        silu1 = conv1 * tf.sigmoid(conv1)
        
        # Split
        s1, s2 = tf.split(silu1, 2, axis=3)
        
        # Transpose on one branch
        t1 = tf.transpose(s1, [0, 3, 1, 2])
        t1 = tf.transpose(t1, [0, 2, 3, 1])
        
        # Reshape on other branch
        r2 = tf.reshape(s2, [1, 320, 320, 16])
        
        # Concat
        concat = tf.concat([t1, r2], axis=3)
        
        # Final conv
        w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), name="w2")
        conv2 = tf.nn.conv2d(concat, w2, strides=[1, 2, 2, 1], padding='SAME')
        
        output = tf.identity(conv2, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        frozen = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        
        with tf.io.gfile.GFile("test_complex.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        print("Created test_complex.pb")

def create_test_depthwise():
    """Test DepthwiseConv2dNative"""
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [1, 640, 640, 3], name="images")
        
        # Regular conv
        w1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), name="w1")
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')
        
        # Depthwise conv
        dw = tf.Variable(tf.random.normal([3, 3, 32, 1]), name="dw")
        depthwise = tf.nn.depthwise_conv2d(conv1, dw, strides=[1, 1, 1, 1], padding='SAME')
        
        # Pointwise conv
        w2 = tf.Variable(tf.random.normal([1, 1, 32, 64]), name="w2")
        conv2 = tf.nn.conv2d(depthwise, w2, strides=[1, 1, 1, 1], padding='SAME')
        
        output = tf.identity(conv2, name="output")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        frozen = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        
        with tf.io.gfile.GFile("test_depthwise.pb", "wb") as f:
            f.write(frozen.SerializeToString())
        print("Created test_depthwise.pb")

if __name__ == "__main__":
    print("Creating test graphs to find the culprit...\n")
    
    create_test_with_transpose()
    create_test_with_reshape()
    create_test_with_split_concat()
    create_test_with_identity_chain()
    create_test_complex()
    create_test_depthwise()
    
    print("\n" + "=" * 70)
    print("Run these tests one by one:")
    print("=" * 70)
    print('''
# Test Transpose
vai_q_tensorflow quantize --input_frozen_graph test_transpose.pb --input_nodes images --input_shapes 1,640,640,3 --output_nodes output --input_fn test_calib.calib_input_yolo --output_dir quant_transpose --calib_iter 1

# Test Reshape  
vai_q_tensorflow quantize --input_frozen_graph test_reshape.pb --input_nodes images --input_shapes 1,640,640,3 --output_nodes output --input_fn test_calib.calib_input_yolo --output_dir quant_reshape --calib_iter 1

# Test Split/Concat
vai_q_tensorflow quantize --input_frozen_graph test_split_concat.pb --input_nodes images --input_shapes 1,640,640,3 --output_nodes output --input_fn test_calib.calib_input_yolo --output_dir quant_split --calib_iter 1

# Test Identity Chain
vai_q_tensorflow quantize --input_frozen_graph test_identity_chain.pb --input_nodes images --input_shapes 1,640,640,3 --output_nodes output --input_fn test_calib.calib_input_yolo --output_dir quant_identity --calib_iter 1

# Test Complex (all combined)
vai_q_tensorflow quantize --input_frozen_graph test_complex.pb --input_nodes images --input_shapes 1,640,640,3 --output_nodes output --input_fn test_calib.calib_input_yolo --output_dir quant_complex --calib_iter 1

# Test Depthwise
vai_q_tensorflow quantize --input_frozen_graph test_depthwise.pb --input_nodes images --input_shapes 1,640,640,3 --output_nodes output --input_fn test_calib.calib_input_yolo --output_dir quant_depthwise --calib_iter 1
''')