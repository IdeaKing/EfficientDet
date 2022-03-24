import collections
import math
import string
from functools import reduce

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow import nn
from tensorflow.keras import backend
from tensorflow.keras import initializers
from tensorflow.keras import layers

MOMENTUM = 0.997
EPSILON = 1e-4
BASE_WEIGHTS_PATH = 'https://github.com/Callidior/keras-applications/releases/download/efficientnet/'

WEIGHTS_HASHES = {'efficientnet-b0': ('163292582f1c6eaca8e7dc7b51b01c61'
                                      '5b0dbc0039699b4dcd0b975cc21533dc',
                                      'c1421ad80a9fc67c2cc4000f666aa507'
                                      '89ce39eedb4e06d531b0c593890ccff3'),
                  'efficientnet-b1': ('d0a71ddf51ef7a0ca425bab32b7fa7f1'
                                      '6043ee598ecee73fc674d9560c8f09b0',
                                      '75de265d03ac52fa74f2f510455ba64f'
                                      '9c7c5fd96dc923cd4bfefa3d680c4b68'),
                  'efficientnet-b2': ('bb5451507a6418a574534aa76a91b106'
                                      'f6b605f3b5dde0b21055694319853086',
                                      '433b60584fafba1ea3de07443b74cfd3'
                                      '2ce004a012020b07ef69e22ba8669333'),
                  'efficientnet-b3': ('03f1fba367f070bd2545f081cfa7f3e7'
                                      '6f5e1aa3b6f4db700f00552901e75ab9',
                                      'c5d42eb6cfae8567b418ad3845cfd63a'
                                      'a48b87f1bd5df8658a49375a9f3135c7'),
                  'efficientnet-b4': ('98852de93f74d9833c8640474b2c698d'
                                      'b45ec60690c75b3bacb1845e907bf94f',
                                      '7942c1407ff1feb34113995864970cd4'
                                      'd9d91ea64877e8d9c38b6c1e0767c411'),
                  'efficientnet-b5': ('30172f1d45f9b8a41352d4219bf930ee'
                                      '3339025fd26ab314a817ba8918fefc7d',
                                      '9d197bc2bfe29165c10a2af8c2ebc675'
                                      '07f5d70456f09e584c71b822941b1952'),
                  'efficientnet-b6': ('f5270466747753485a082092ac9939ca'
                                      'a546eb3f09edca6d6fff842cad938720',
                                      '1d0923bb038f2f8060faaf0a0449db4b'
                                      '96549a881747b7c7678724ac79f427ed'),
                  'efficientnet-b7': ('876a41319980638fa597acbbf956a82d'
                                      '10819531ff2dcb1a52277f10c7aefa1a',
                                      '60b56ff3a8daccc8d96edfd40b204c11'
                                      '3e51748da657afd58034d54d3cec2bac')}

BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size',
                                                 'num_repeat',
                                                 'input_filters',
                                                 'output_filters',
                                                 'expand_ratio',
                                                 'id_skip',
                                                 'strides',
                                                 'se_ratio'])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                                 expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
                       BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                                 expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                       BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                                 expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                       BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                                 expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                       BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                                 expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
                       BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                                 expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
                       BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                                 expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)]

CONV_KERNEL_INITIALIZER = {'class_name': 'VarianceScaling',
                           'config': {'scale': 2.0, 'mode': 'fan_out', 'distribution': 'normal'}}

DENSE_KERNEL_INITIALIZER = {'class_name': 'VarianceScaling',
                            'config': {'scale': 1. / 3., 'mode': 'fan_out', 'distribution': 'uniform'}}


def swish(x):
    return nn.swish(x)


class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix=''):
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = FixedDropout(drop_rate,
                             noise_shape=(None, 1, 1, 1),
                             name=prefix + 'drop')(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def efficient_net(width_coefficient,
                  depth_coefficient,
                  drop_connect_rate=0.2,
                  depth_divisor=8,
                  blocks_args=None,
                  input_tensor=None):
    if blocks_args is None:
        blocks_args = DEFAULT_BLOCKS_ARGS
    features = []

    img_input = input_tensor

    bn_axis = 3
    activation = swish

    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        block_args = block_args._replace(input_filters=round_filters(block_args.input_filters,
                                                                     width_coefficient,
                                                                     depth_divisor),
                                         output_filters=round_filters(block_args.output_filters,
                                                                      width_coefficient,
                                                                      depth_divisor),
                                         num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            for b_idx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1,
                                                   string.ascii_lowercase[b_idx + 1])
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1
        if idx < len(blocks_args) - 1 and blocks_args[idx + 1].strides[0] == 2:
            features.append(x)
        elif idx == len(blocks_args) - 1:
            features.append(x)
    return features


def efficient_net_b0(input_tensor=None):
    return efficient_net(1.0, 1.0, input_tensor=input_tensor)


def efficient_net_b1(input_tensor=None):
    return efficient_net(1.0, 1.1, input_tensor=input_tensor)


def efficient_net_b2(input_tensor=None):
    return efficient_net(1.1, 1.2, input_tensor=input_tensor)


def efficient_net_b3(input_tensor=None):
    return efficient_net(1.2, 1.4, input_tensor=input_tensor)


def efficient_net_b4(input_tensor=None):
    return efficient_net(1.4, 1.8, input_tensor=input_tensor)


def efficient_net_b5(input_tensor=None):
    return efficient_net(1.6, 2.2, input_tensor=input_tensor)


def efficient_net_b6(input_tensor=None):
    return efficient_net(1.8, 2.6, input_tensor=input_tensor)


def efficient_net_b7(input_tensor=None):
    return efficient_net(2.0, 3.1, input_tensor=input_tensor)


def separable_convolution(num_channels, kernel_size, strides, name):
    f1 = layers.SeparableConv2D(num_channels, kernel_size, strides, 'same', name=f'{name}_conv')
    f2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}_bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def build_fpn(features, num_channels, index):
    if index == 0:
        _, _, c3, c4, c5 = features
        p3_in = c3
        p4_in = c4
        p5_in = c5
        p6_in = layers.Conv2D(num_channels, 1, 1, 'same', name=f'resample_p6_conv_{index}')(c5)
        p6_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'resample_p6_bn_{index}')(p6_in)
        p6_in = layers.MaxPooling2D(3, 2, 'same', name=f'resample_p6_pool_{index}')(p6_in)
        p7_in = layers.MaxPooling2D(3, 2, 'same', name=f'resample_p7_pool_{index}')(p6_in)
        p7_u = layers.UpSampling2D(name=f'fpn_upsample1_{index}')(p7_in)
        p6_td = layers.Add(name=f'fpn_add1_{index}')([p6_in, p7_u])
        p6_td = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation1_{index}')(p6_td)
        p6_td = separable_convolution(num_channels, 3, 1, name=f'fpn_conv1_{index}')(p6_td)
        p5_in_1 = layers.Conv2D(num_channels, 1, 1, 'same', name=f'fpn_conv2_{index}')(p5_in)
        p5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_bn1_{index}')(p5_in_1)
        p6_u = layers.UpSampling2D(name=f'fpn_upsample2_{index}')(p6_td)
        p5_td = layers.Add(name=f'fpn_add2_{index}')([p5_in_1, p6_u])
        p5_td = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation2_{index}')(p5_td)
        p5_td = separable_convolution(num_channels, 3, 1, name=f'fpn_conv3_{index}')(p5_td)
        p4_in_1 = layers.Conv2D(num_channels, 1, 1, 'same', name=f'fpn_conv4_{index}')(p4_in)
        p4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_bn2_{index}')(p4_in_1)
        p5_u = layers.UpSampling2D(name=f'fpn_upsample3_{index}')(p5_td)
        p4_td = layers.Add(name=f'fpn_add3_{index}')([p4_in_1, p5_u])
        p4_td = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation3_{index}')(p4_td)
        p4_td = separable_convolution(num_channels, 3, 1, name=f'fpn_conv5_{index}')(p4_td)
        p3_in = layers.Conv2D(num_channels, 1, 1, 'same', name=f'fpn_conv6_{index}')(p3_in)
        p3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_bn3_{index}')(p3_in)
        p4_u = layers.UpSampling2D(name=f'fpn_upsample4_{index}')(p4_td)
        p3_out = layers.Add(name=f'fpn_add4_{index}')([p3_in, p4_u])
        p3_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation4_{index}')(p3_out)
        p3_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv7_{index}')(p3_out)
        p4_in_2 = layers.Conv2D(num_channels, 1, 1, 'same', name=f'fpn_conv8_{index}')(p4_in)
        p4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_bn4_{index}')(p4_in_2)
        p3_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool1_{index}')(p3_out)
        p4_out = layers.Add(name=f'fpn_add5_{index}')([p4_in_2, p4_td, p3_d])
        p4_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation5_{index}')(p4_out)
        p4_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv9_{index}')(p4_out)

        p5_in_2 = layers.Conv2D(num_channels, 1, 1, 'same', name=f'fpn_conv10_{index}')(p5_in)
        p5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_bn5_{index}')(p5_in_2)
        p4_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool2_{index}')(p4_out)
        p5_out = layers.Add(name=f'fpn_add6_{index}')([p5_in_2, p5_td, p4_d])
        p5_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation6_{index}')(p5_out)
        p5_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv11_{index}')(p5_out)

        p5_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool3_{index}')(p5_out)
        p6_out = layers.Add(name=f'fpn_add7_{index}')([p6_in, p6_td, p5_d])
        p6_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation7_{index}')(p6_out)
        p6_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv12_{index}')(p6_out)

        p6_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool4_{index}')(p6_out)
        p7_out = layers.Add(name=f'fpn_add8_{index}')([p7_in, p6_d])
        p7_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation8_{index}')(p7_out)
        p7_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv13_{index}')(p7_out)

    else:
        p3_in, p4_in, p5_in, p6_in, p7_in = features
        p7_u = layers.UpSampling2D(name=f'fpn_upsample1_{index}')(p7_in)
        p6_td = layers.Add(name=f'fpn_add1_{index}')([p6_in, p7_u])
        p6_td = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation1_{index}')(p6_td)
        p6_td = separable_convolution(num_channels, 3, 1, name=f'fpn_conv1_{index}')(p6_td)
        p6_u = layers.UpSampling2D(name=f'fpn_upsample2_{index}')(p6_td)
        p5_td = layers.Add(name=f'fpn_add2_{index}')([p5_in, p6_u])
        p5_td = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation2_{index}')(p5_td)
        p5_td = separable_convolution(num_channels, 3, 1, name=f'fpn_conv2_{index}')(p5_td)
        p5_u = layers.UpSampling2D(name=f'fpn_upsample3_{index}')(p5_td)
        p4_td = layers.Add(name=f'fpn_add3_{index}')([p4_in, p5_u])
        p4_td = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation3_{index}')(p4_td)
        p4_td = separable_convolution(num_channels, 3, 1, name=f'fpn_conv3_{index}')(p4_td)
        p4_u = layers.UpSampling2D(name=f'fpn_upsample4_{index}')(p4_td)
        p3_out = layers.Add(name=f'fpn_add4_{index}')([p3_in, p4_u])
        p3_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation4_{index}')(p3_out)
        p3_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv4_{index}')(p3_out)
        p3_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool1_{index}')(p3_out)
        p4_out = layers.Add(name=f'fpn_add5_{index}')([p4_in, p4_td, p3_d])
        p4_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation5_{index}')(p4_out)
        p4_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv5_{index}')(p4_out)

        p4_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool2_{index}')(p4_out)
        p5_out = layers.Add(name=f'fpn_add6_{index}')([p5_in, p5_td, p4_d])
        p5_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation6_{index}')(p5_out)
        p5_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv6_{index}')(p5_out)

        p5_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool3_{index}')(p5_out)
        p6_out = layers.Add(name=f'fpn_add7_{index}')([p6_in, p6_td, p5_d])
        p6_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation7_{index}')(p6_out)
        p6_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv7_{index}')(p6_out)

        p6_d = layers.MaxPooling2D(3, 2, 'same', name=f'fpn_pool4_{index}')(p6_out)
        p7_out = layers.Add(name=f'fpn_add8_{index}')([p7_in, p6_d])
        p7_out = layers.Activation(lambda x: nn.swish(x), name=f'fpn_activation8_{index}')(p7_out)
        p7_out = separable_convolution(num_channels, 3, 1, name=f'fpn_conv8_{index}')(p7_out)
    return p3_out, p4_td, p5_td, p6_td, p7_out


class PriorProbability(initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {'probability': self.probability}

    def __call__(self, shape, dtype=None):
        result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

        return result


class ClipBoxes(layers.Layer):
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = backend.cast(backend.shape(image), backend.floatx())
        height = shape[1]
        width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        return backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class RegressBoxes(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        cxa = (anchors[..., 0] + anchors[..., 2]) / 2
        cya = (anchors[..., 1] + anchors[..., 3]) / 2
        wa = anchors[..., 2] - anchors[..., 0]
        ha = anchors[..., 3] - anchors[..., 1]
        ty, tx, th, tw = regression[..., 0], regression[..., 1], regression[..., 2], regression[..., 3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        cy = ty * ha + cya
        cx = tx * wa + cxa
        y_min = cy - h / 2.
        x_min = cx - w / 2.
        y_max = cy + h / 2.
        x_max = cx + w / 2.
        return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        return config


class FilterDetections(layers.Layer):
    def __init__(self,
                 nms=True,
                 class_specific_filter=True,
                 nms_threshold=0.1,
                 score_threshold=0.01,
                 max_detections=100,
                 parallel_iterations=32,
                 detect_quadrangle=False,
                 **kwargs):
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.detect_quadrangle = detect_quadrangle
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        boxes = inputs[0]
        classification = inputs[1]

        def filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]

            def filter_detection(scores_, labels_):
                indices_ = tf.where(backend.greater(scores_, self.score_threshold))

                if self.nms:
                    filtered_boxes = tf.gather_nd(boxes_, indices_)
                    filtered_scores = backend.gather(scores_, indices_)[:, 0]

                    nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, self.max_detections,
                                                               0.1)
                    indices_ = backend.gather(indices_, nms_indices)

                labels_ = tf.gather_nd(labels_, indices_)
                indices_ = backend.stack([indices_[:, 0], labels_], axis=1)

                return indices_

            if self.class_specific_filter:
                all_indices = []
                for c in range(int(classification_.shape[1])):
                    scores = classification_[:, c]
                    labels = c * tf.ones((backend.shape(scores)[0],), dtype='int64')
                    all_indices.append(filter_detection(scores, labels))
                indices = backend.concatenate(all_indices, axis=0)
            else:
                scores = backend.max(classification_, axis=1)
                labels = backend.argmax(classification_, axis=1)
                indices = filter_detection(scores, labels)

            scores = tf.gather_nd(classification_, indices)
            labels = indices[:, 1]
            scores, top_indices = tf.nn.top_k(scores, k=backend.minimum(self.max_detections, backend.shape(scores)[0]))

            indices = backend.gather(indices[:, 0], top_indices)
            boxes_ = backend.gather(boxes_, indices)
            labels = backend.gather(labels, top_indices)

            pad_size = backend.maximum(0, self.max_detections - backend.shape(scores)[0])
            boxes_ = tf.pad(boxes_, [[0, pad_size], [0, 0]], constant_values=-1)
            scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
            labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
            labels = backend.cast(labels, 'int32')

            boxes_.set_shape([self.max_detections, 4])
            scores.set_shape([self.max_detections])
            labels.set_shape([self.max_detections])

            return [boxes_, scores, labels]

        outputs = tf.map_fn(fn=filter_detections,
                            elems=[boxes, classification],
                            dtype=['float32', 'float32', 'int32'],
                            parallel_iterations=self.parallel_iterations)

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections), ]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs) + 1) * [None]

    def get_config(self):
        layer_config = super(FilterDetections, self).get_config()
        layer_config.update({'nms': self.nms,
                             'class_specific_filter': self.class_specific_filter,
                             'nms_threshold': self.nms_threshold,
                             'score_threshold': self.score_threshold,
                             'max_detections': self.max_detections,
                             'parallel_iterations': self.parallel_iterations, })

        return layer_config
