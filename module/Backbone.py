import tensorflow as tf
from tensorflow.contrib import slim
from module.Transformer_Modules import positional_encoding, ff, multihead_attention

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def mean_image_subtraction(images, means=(128.0, 128.0, 128.0)):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    with tf.variable_scope("mean_subtraction"):
        num_channels = images.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
        for i in range(num_channels):
            channels[i] -= means[i]
            return tf.concat(axis=3, values=channels)

class Backbone(object):
    def __init__(self, weight_decay=1e-5, is_training=True):
        self.weight_decay = weight_decay
        self.is_training = is_training
    def shortcut(self, inputs, output_dim, stride=1):
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if stride == [1, 1] and depth_in == output_dim:
            shortcut = inputs
        else:
            shortcut = slim.conv2d(inputs, output_dim, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        return shortcut

    def basicblock(self, inputs, mid_dim, output_dim, stride, scope):
        with tf.variable_scope(scope):
            shortcut = self.shortcut(inputs, output_dim, stride)
            residual = slim.conv2d(inputs, mid_dim, 1, stride=stride, scope='conv1')
            residual = slim.conv2d(residual, mid_dim, 3, stride=1, scope='conv2')
            residual = slim.conv2d(residual, output_dim, 1, stride=1, activation_fn=None, scope='conv3')

            return tf.nn.relu(shortcut + residual)

    def resnet(self, input_image):
        with tf.variable_scope("resnet"):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': self.is_training
            }
            end_points = {}
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                net = slim.conv2d(input_image, 64, 7, stride=2, scope='conv1') # 32 * 128 * 64
                net = slim.max_pool2d(net, kernel_size=3, stride=2, scope='pool1') # 16 * 64 * 64
                # stage 1: 3 x blocks
                with tf.variable_scope("Block1"):
                    net = self.basicblock(net, 64, 256, [1, 1], scope="Block1_1")
                    net = self.basicblock(net, 64, 256, [1, 1], scope="Block1_2")
                    net = self.basicblock(net, 64, 256, [1, 1], scope="Block1_3") # 16 * 64 * 32
                    end_points['pool2'] = net
                # stage 2: 4 x blocks
                with tf.variable_scope("Block2"):
                    net = self.basicblock(net, 128, 512, [2, 2], scope="Block2_1")
                    net = self.basicblock(net, 128, 512, [1, 1], scope="Block2_2")
                    net = self.basicblock(net, 128, 512, [1, 1], scope="Block2_3")
                    net = self.basicblock(net, 128, 512, [1, 1], scope="Block2_4") # 8 * 32 * 512
                    end_points['pool3'] = net
                # stage 3: 6 x blocks
                with tf.variable_scope("Block3"):
                    net = self.basicblock(net, 256, 1024, [2, 2], scope="Block3_1")
                    net = self.basicblock(net, 256, 1024, [1, 1], scope="Block3_2")
                    net = self.basicblock(net, 256, 1024, [1, 1], scope="Block3_3")
                    net = self.basicblock(net, 256, 1024, [1, 1], scope="Block3_4")
                    net = self.basicblock(net, 256, 1024, [1, 1], scope="Block3_5")
                    net = self.basicblock(net, 256, 1024, [1, 1], scope="Block3_6") # 4 * 16 * 1024
                    end_points['pool4'] = net
                # stage 4: 3 x blocks
                with tf.variable_scope("Block5"):
                    net = self.basicblock(net, 512, 2048, [2, 2], scope="Block5_1")
                    net = self.basicblock(net, 512, 2048, [1, 1], scope="Block5_2")
                    net = self.basicblock(net, 512, 2048, [1, 1], scope="Block5_3") # 2 * 8 * 2048
                    end_points['pool5'] = net
        return end_points

    def transformer_units(self, feature_map):
        with tf.variable_scope("transformer_units"):
            N, H, W, C = feature_map.shape.as_list()
            enc_feature_map = tf.reshape(feature_map, [N, (H * W), C])
            loc_embed = positional_encoding(enc_feature_map, maxlen=(H * W))
            enc_feature_map += loc_embed

            enc_feature_map, _ = multihead_attention(enc_feature_map, enc_feature_map, enc_feature_map, num_heads=8, dropout_rate=0.1, training=self.is_training, scope="multi_head_attention_1")
            enc_feature_map = ff(enc_feature_map, num_units=(512, 512), scope="ff_1")
            enc_feature_map, _ = multihead_attention(enc_feature_map, enc_feature_map, enc_feature_map, num_heads=8, dropout_rate=0.1, training=self.is_training, scope="multi_head_attention_2")
            enc_feature_map = ff(enc_feature_map, num_units=(512, 512), scope="ff_2")

            enc_feature = tf.reshape(enc_feature_map, [N, H, W, -1])

            return enc_feature

    def __call__(self, input_image):
        # images = mean_image_subtraction(input_image)
        images = input_image
        with tf.variable_scope("fpn"):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': self.is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                end_points = self.resnet(images)
                f = [end_points['pool5'], end_points['pool4'], end_points['pool3']]
                for i in range(3):  # 4
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None]
                h = [None, None, None]
                num_outputs = [None, 1024, 512]
                for i in range(3):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 1:
                        g[i] = unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
            feature_map = g[2]
            tu_feature_map = self.transformer_units(feature_map)
            return feature_map, tu_feature_map