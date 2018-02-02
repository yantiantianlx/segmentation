from nets.layers import *
import tensorflow as tf

def residual_inference(images,training=True):
    print('-'*30)
    print('Network Architecture')
    print('-'*30)
    layer_name_dict = {}
    def layer_name(base_name):
        if base_name not in layer_name_dict:
            layer_name_dict[base_name] = 0
        layer_name_dict[base_name] += 1
        name = base_name + str(layer_name_dict[base_name])
        return name

    NUM_CLASS = 14
    bn = True
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    x = images

    # Build and return the network
    for i in range(4):
        x = conv_layer(x,layer_name('conv'),3,64,he_initializer,bn=bn,training=training)
    x = residual_block(x,layer_name('resblock'),3,64,he_initializer,stride=2,bn=bn,training=training)
    for i in range(8):
        x = residual_block(x,layer_name('resblock'),3,64,he_initializer,bn=bn,training=training)
    x = residual_block(x,layer_name('resblock'),3,128,he_initializer,stride=2,bn=bn,training=training)
    for i in range(16):
        x = residual_block(x,layer_name('resblock'),3,128,he_initializer,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,128,he_initializer,stride=2,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,64,he_initializer,stride=2,bn=bn,training=training)
    x = deconv_layer(x,layer_name('deconv'),3,NUM_CLASS,he_initializer,bn=False,training=training,relu=False)
    print('-'*30)
    print('')
    return x