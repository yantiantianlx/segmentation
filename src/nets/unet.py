from nets.layers import *
import tensorflow as tf
from collections import OrderedDict

def unet(input, initializer, num_classes = 2, training=True, bn = True):
    print('-' * 30)
    print('Network Architecture')
    print('-' * 30)
    layer_name_dict = {}

    def layer_name(base_name):
        if base_name not in layer_name_dict:
            layer_name_dict[base_name] = 0
        layer_name_dict[base_name] += 1
        name = base_name + str(layer_name_dict[base_name])
        return name

    x = input

    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    # Build the network
    x = conv_layer(x, layer_name('conv'), 3, 64, initializer, bn=bn, training=training)
    dw_h_convs[0] = conv_layer(x, layer_name('conv'), 3, 64, initializer, bn=bn, training=training)
    x = max_pooling(dw_h_convs[0], 'pool1')

    dw_h_convs[1] = conv_layer(x, layer_name('conv'), 3, 128, initializer, bn=bn, training=training)
    dw_h_convs[1] = conv_layer(dw_h_convs[1], layer_name('conv'), 3, 128, initializer, bn=bn, training=training)
    dw_h_convs[2] = max_pooling(dw_h_convs[1], 'pool2')

    dw_h_convs[2] = conv_layer(dw_h_convs[2], layer_name('conv'), 3, 256, initializer, bn=bn, training=training)
    dw_h_convs[2] = conv_layer(dw_h_convs[2], layer_name('conv'), 3, 256, initializer, bn=bn, training=training)
    dw_h_convs[3] = max_pooling(dw_h_convs[2], 'pool3')

    dw_h_convs[3] = conv_layer(dw_h_convs[3], layer_name('conv'), 3, 512, initializer, bn=bn, training=training)
    dw_h_convs[3] = conv_layer(dw_h_convs[3], layer_name('conv'), 3, 512, initializer, bn=bn, training=training)
    dw_h_convs[4] = max_pooling(dw_h_convs[3], 'pool4')

    dw_h_convs[4] = conv_layer(dw_h_convs[4], layer_name('conv'), 3, 1024, initializer, bn=bn, training=training)
    dw_h_convs[4] = conv_layer(dw_h_convs[4], layer_name('conv'), 3, 512, initializer, bn=bn, training=training)

    up_h_convs[0] = tf.image.resize_images(dw_h_convs[4], [dw_h_convs[3].get_shape().as_list()[1],
                                                           dw_h_convs[3].get_shape().as_list()[2]])

    up_h_convs[0] = tf.concat([up_h_convs[0], dw_h_convs[3]], 3)
    up_h_convs[0] = conv_layer(up_h_convs[0], layer_name('conv'), 3, 512, initializer, bn=bn, training=training)
    up_h_convs[0] = conv_layer(up_h_convs[0], layer_name('conv'), 3, 256, initializer, bn=bn, training=training)

    up_h_convs[1] = tf.image.resize_images(up_h_convs[0], [dw_h_convs[2].get_shape().as_list()[1],
                                                           dw_h_convs[2].get_shape().as_list()[2]])

    up_h_convs[1] = tf.concat([up_h_convs[1], dw_h_convs[2]], 3)
    up_h_convs[1] = conv_layer(up_h_convs[1], layer_name('conv'), 3, 256, initializer, bn=bn, training=training)
    up_h_convs[1] = conv_layer(up_h_convs[1], layer_name('conv'), 3, 128, initializer, bn=bn, training=training)

    up_h_convs[2] = tf.image.resize_images(up_h_convs[1], [dw_h_convs[1].get_shape().as_list()[1],
                                                           dw_h_convs[1].get_shape().as_list()[2]])

    up_h_convs[2] = tf.concat([up_h_convs[2], dw_h_convs[1]], 3)
    up_h_convs[2] = conv_layer(up_h_convs[2], layer_name('conv'), 3, 128, initializer, bn=bn, training=training)
    up_h_convs[2] = conv_layer(up_h_convs[2], layer_name('conv'), 3, 64, initializer, bn=bn, training=training)

    up_h_convs[3] = tf.image.resize_images(up_h_convs[2], [dw_h_convs[0].get_shape().as_list()[1],
                                                           dw_h_convs[0].get_shape().as_list()[2]])

    up_h_convs[3] = tf.concat([up_h_convs[3], dw_h_convs[0]], 3)
    up_h_convs[3] = conv_layer(up_h_convs[3], layer_name('conv'), 3, 64, initializer, bn=bn, training=training)
    up_h_convs[3] = conv_layer(up_h_convs[3], layer_name('conv'), 3, 64, initializer, bn=bn, training=training)

    out = conv_layer(up_h_convs[3], layer_name('conv'), 1, num_classes, initializer, bn=False, training=training, relu=False)

    print('size of out= ', out.get_shape().as_list())
    return out