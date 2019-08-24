import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32

# Model
def convolution(x, train=False):
    """The Model definition."""
    conv1_w = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED, dtype=data_type()))
    conv1_b = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv2_w = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=data_type()))
    conv2_b = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))

    fc1_w = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    fc1_b = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))

    fc2_w = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                            stddev=0.1,
                                            seed=SEED,
                                            dtype=data_type()))
    fc2_b = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=data_type()))

    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    data = tf.reshape(x, [-1, 28, 28, 1])
    conv = tf.nn.conv2d(data,
                        conv1_w,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_b))
    # Max pooling. The kernel size sp {ksize} also follows the layout of
    # the data. Here we have a poolin of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_w,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_b))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_w) + fc1_b)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    y = tf.nn.softmax(tf.matmul(hidden, fc2_w) + fc2_b)
    return y, [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b]

