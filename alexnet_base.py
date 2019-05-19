"""
Created Date: May 09, 2019

Created By: varunravivarma
-------------------------------------------------------------------------------

alexnet_base.py:
Create a base alexnet model to test outputs and accuracy on character classification.
"""

import tensorflow as tf
import numpy as np
import math
import img_augment as ia

input_shape = 70 * 70 * 3

n_classes = 29

# input and output vector placeholders
x = tf.placeholder(tf.float32, [None, input_shape])
y = tf.placeholder(tf.float32, [None, n_classes])

def fc_layer(x, W, b, name=None): return tf.nn.bias_add(tf.matmul(x, W), b)


def alexnet(image, weights, biases):
    # Reshape image to 70x70x3
    image = tf.reshape(image, [-1, 70, 70, 3])

    # Convolution 1
    conv1 = tf.nn.conv2d(image, weights["wt_conv_1"], strides=[
                         1, 3, 3, 1], padding="SAME", name="conv1")
    conv1 = tf.nn.bias_add(conv1, biases["bias_conv_1"])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.local_response_normalization(
        conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    # Convolution 2
    conv2 = tf.nn.conv2d(conv1, weights["wt_conv_2"], strides=[
                         1, 1, 1, 1], padding="SAME", name="conv2")
    conv2 = tf.nn.bias_add(conv2, biases["bias_conv_2"])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.local_response_normalization(
        conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

    # Convolution 3
    conv3 = tf.nn.conv2d(conv2, weights["wt_conv_3"], strides=[
                         1, 1, 1, 1], padding="SAME", name="conv3")
    conv3 = tf.nn.bias_add(conv3, biases["bias_conv_3"])
    conv3 = tf.nn.relu(conv3)

    # Convolution 4
    conv4 = tf.nn.conv2d(conv3, weights["wt_conv_4"], strides=[
                         1, 1, 1, 1], padding="SAME", name="conv4")
    conv4 = tf.nn.bias_add(conv4, biases["bias_conv_4"])
    conv4 = tf.nn.relu(conv4)

    # Convolution 5
    conv5 = tf.nn.conv2d(conv4, weights["wt_conv_5"], strides=[
                         1, 1, 1, 1], padding="SAME", name="conv5")
    conv5 = tf.nn.bias_add(conv5, biases["bias_conv_5"])
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[
                           1, 2, 2, 1], padding="VALID")

    # Reshape and flatten Convolution 5
    shape = [-1, weights['wt_fc_1'].get_shape().as_list()[0]]
    flatten = tf.reshape(conv5, shape)

    # Fully connected 1
    fc1 = fc_layer(flatten, weights["wt_fc_1"], biases["bias_fc_1"], name="fc1")
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    # Fully connected 2
    fc2 = fc_layer(fc1, weights["wt_fc_2"], biases["bias_fc_2"], name="fc2")
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    # Fully connected 3
    fc3 = fc_layer(fc2, weights["wt_fc_3"], biases["bias_fc_3"], name="fc3")
    fc3 = tf.nn.softmax(fc3)

    return fc3


# Weight parameters as devised in the original research paper
weights = {
        "wt_conv_1": tf.Variable(tf.truncated_normal([11, 11, 3, 96],     stddev=0.01), name="wt_conv_1"),
        "wt_conv_2": tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name="wt_conv_2"),
        "wt_conv_3": tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name="wt_conv_3"),
        "wt_conv_4": tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name="wt_conv_4"),
        "wt_conv_5": tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name="wt_conv_5"),
        "wt_fc_1": tf.Variable(tf.truncated_normal([28*28*256, 4096],   stddev=0.01), name="wt_fc_1"),
        "wt_fc_2": tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name="wt_fc_2"),
        "wt_fc_3": tf.Variable(tf.truncated_normal([4096, n_classes],   stddev=0.01), name="wt_fc_3")
}

# Bias parameters as devised in the original research paper
biases = {
        "bias_conv_1": tf.Variable(tf.constant(0.0, shape=[96]),        name="bias_conv_1"),
        "bias_conv_2": tf.Variable(tf.constant(1.0, shape=[256]),       name="bias_conv_2"),
        "bias_conv_3": tf.Variable(tf.constant(0.0, shape=[384]),       name="bias_conv_3"),
        "bias_conv_4": tf.Variable(tf.constant(1.0, shape=[384]),       name="bias_conv_4"),
        "bias_conv_5": tf.Variable(tf.constant(1.0, shape=[256]),       name="bias_conv_5"),
        "bias_fc_1": tf.Variable(tf.constant(1.0, shape=[4096]),      name="bias_fc_1"),
        "bias_fc_2": tf.Variable(tf.constant(1.0, shape=[4096]),      name="bias_fc_2"),
        "bias_fc_3": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bias_fc_3")
}
