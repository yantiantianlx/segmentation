import tensorflow as tf

DATA_TYPE = tf.float32

def variable(name, shape, initializer, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=DATA_TYPE, trainable=True)

def batch_norm_layer(input_tensor, scope, training):
    return tf.contrib.layers.batch_norm(input_tensor, scope=scope, is_training=training, decay=0.99)

def conv_layer(input_tensor, name, kernel_size, output_channels, initializer, stride=1, bn=False, training=False, relu=True):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer,
                          regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
        conv_layer = tf.nn.bias_add(conv, biases)
        if bn:
            conv_layer = batch_norm_layer(conv_layer, scope, training)
        if relu:
            conv_layer = tf.nn.relu(conv_layer, name=scope.name)
    print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), conv_layer.get_shape().as_list()))
    return conv_layer

def residual_block(input_tensor, name, kernel_size, output_channels, initializer, stride=1, bn=True, training=False):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        conv_output = conv_layer(input_tensor, 'conv1', kernel_size, output_channels, initializer, stride=stride, bn=bn,
                                 training=training, relu=True)
        conv_output = conv_layer(conv_output, 'conv2', kernel_size, output_channels, initializer, stride=1, bn=bn,
                                 training=training, relu=False)
        if stride != 1 or input_channels != output_channels:
            old_input_shape = input_tensor.get_shape().as_list()
            input_tensor = conv_layer(input_tensor, 'projection', stride, output_channels, initializer, stride=stride,
                                      bn=False, training=training, relu=False)
            print('Projecting input {0} -> {1}'.format(old_input_shape, input_tensor.get_shape().as_list()))
        res_output = tf.nn.relu(input_tensor + conv_output, name=scope.name)
    return res_output

def deconv_layer(input_tensor, name, kernel_size, output_channels, initializer, stride=1, bn=False, training=False, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    output_shape = list(input_shape)
    output_shape[1] *= stride
    output_shape[2] *= stride
    output_shape[3] = output_channels
    with tf.variable_scope(name) as scope:
        kernel = variable('weights', [kernel_size, kernel_size, output_channels, input_channels], initializer,
                          regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input_tensor, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
        deconv_layer = tf.nn.bias_add(deconv, biases)
        if bn:
            deconv_layer = batch_norm_layer(deconv_layer, scope, training)
        if relu:
            deconv_layer = tf.nn.relu(deconv_layer, name=scope.name)
    print('Deconv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), deconv_layer.get_shape().as_list()))
    return deconv_layer

def max_pooling(input_tensor, name, kernel = 3, factor=2):
    pool = tf.nn.max_pool(input_tensor, ksize=[1, kernel, kernel, 1], strides=[1, factor, factor, 1], padding='SAME',name=name)
    print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), pool.get_shape().as_list()))
    return pool

def fully_connected_layer(input_tensor, name, output_channels, initializer, bn=False, training=False, relu=True):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        weights = variable('weights', [input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(input_tensor, weights), biases, name=scope.name)
        if bn:
            fc = batch_norm_layer(fc, scope, training)
        if relu:
            fc = tf.nn.relu(fc, name=scope.name)
    print('Fully connected layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), fc.get_shape().as_list()))
    return fc

def dropout_layer(input_tensor, keep_prob, training):
    if training:
        return tf.nn.dropout(input_tensor, keep_prob)
    return input_tensor

def concat_layer(input_tensor1, input_tensor2, axis=3):
    output = tf.concat(axis, [input_tensor1, input_tensor2])
    input1_shape = input_tensor1.get_shape().as_list()
    input2_shape = input_tensor2.get_shape().as_list()
    output_shape = output.get_shape().as_list()
    print('Concat layer {0} and {1} -> {2}'.format(input1_shape, input2_shape, output_shape))
    return output

def flatten(input_tensor, name):
    batch_size = input_tensor.get_shape().as_list()[0]
    with tf.variable_scope(name) as scope:
        flat = tf.reshape(input_tensor, [batch_size, -1])
    print('Flatten layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), flat.get_shape().as_list()))
    return flat

def loss(predictions, labels):
    # predictions : [b, h, w, num_class], float32 or int32
    # labels : [b, h, w, num_class] or [b, h * w * num_class], int32
    # out : loss + weight_loss
    num_classes = predictions.get_shape().as_list()[-1]
    flat_predictions = tf.reshape(predictions, [-1, num_classes])
    flat_labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_predictions, labels=flat_labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    weight_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return tf.add(cross_entropy_mean, weight_loss)

def accuracy(predictions, labels):
    # predictions : [b, h, w, num_class], float32 or int32
    # labels : [b, h, w, num_class] or [b, h * w * num_class], int32
    # out : [1], float
    batch_size = predictions.get_shape().as_list()[0]
    arg_max_preds = tf.argmax(predictions, 3)
    flat_predictions = tf.reshape(arg_max_preds, [batch_size, -1])
    flat_labels = tf.reshape(labels, [batch_size, -1])
    correct_prediction = tf.equal(tf.cast(flat_predictions, tf.int32), flat_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
