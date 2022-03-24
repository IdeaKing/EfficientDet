import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, backend, layers, models

from nets.helper import ClipBoxes, RegressBoxes, FilterDetections, PriorProbability
from nets.helper import build_fpn
from nets.helper import efficient_net_b0, efficient_net_b1, efficient_net_b2
from nets.helper import efficient_net_b3, efficient_net_b4, efficient_net_b5, efficient_net_b6
from utils import config
from utils.util import anchors_for_shape

w_bi_fpns = [64, 88, 112, 160, 224, 288, 384]
d_bi_fpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
backbones = [efficient_net_b0, efficient_net_b1, efficient_net_b2,
             efficient_net_b3, efficient_net_b4, efficient_net_b5, efficient_net_b6]

MOMENTUM = 0.997
EPSILON = 1e-4


class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, separable_conv=True, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        num_values = 4
        options = {'kernel_size': 3,
                   'strides': 1,
                   'padding': 'same',
                   'bias_initializer': 'zeros', }
        if separable_conv:
            kernel_initializer = {'depthwise_initializer': initializers.VarianceScaling(),
                                  'pointwise_initializer': initializers.VarianceScaling(), }
            options.update(kernel_initializer)
            self.conv = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in
                         range(depth)]
            self.head = layers.SeparableConv2D(filters=num_anchors * num_values,
                                               name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)}
            options.update(kernel_initializer)
            self.conv = [layers.Conv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.Conv2D(filters=num_anchors * num_values, name=f'{self.name}/box-predict', **options)

        self.bn = [[layers.BatchNormalization(-1, MOMENTUM, EPSILON, name=f'{self.name}/box-{i}-bn-{j}') for j in
                    range(3, 8)] for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_values))
        self.level = 0

    def call(self, inputs, **kwargs):
        feature = inputs
        for i in range(self.depth):
            # print("box net, i", i, " levle", self.level)
            feature = self.conv[i](feature)
            feature = self.bn[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        if self.level == self.depth:
            self.level = 0
        return outputs


class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {'kernel_size': 3, 'strides': 1, 'padding': 'same', }
        if self.separable_conv:
            kernel_initializer = {'depthwise_initializer': initializers.VarianceScaling(),
                                  'pointwise_initializer': initializers.VarianceScaling(), }
            options.update(kernel_initializer)
            self.conv = [layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                **options)
                         for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)}
            options.update(kernel_initializer)
            self.conv = [layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                       **options)
                         for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}') for j
             in range(3, 8)]
            for i in range(depth)]
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        # print("self level", self.level)
        # print("self depth", self.depth)
        feature = inputs
        for i in range(self.depth):
            feature = self.conv[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        self.level += 1
        if self.level == self.depth:
            self.level = 0

        return outputs

def build_model(phi, num_classes=20, num_anchors=9, score_threshold=0.01, anchor_parameters=None, separable_conv=True):
    assert phi in range(7)
    input_size = config.image_size
    input_shape = (input_size, input_size, 3)
    image_input = layers.Input(input_shape)
    w_bi_fpn = w_bi_fpns[phi]
    d_bi_fpn = d_bi_fpns[phi]
    w_head = w_bi_fpn
    d_head = d_heads[phi]
    backbone_cls = backbones[phi]
    features = backbone_cls(input_tensor=image_input)
    fpn_features = features
    for i in range(d_bi_fpn):
        fpn_features = build_fpn(fpn_features, w_bi_fpn, i)
    box_net = BoxNet(w_head, d_head, num_anchors, separable_conv, name='box_net')
    class_net = ClassNet(w_head, d_head, num_classes, num_anchors, separable_conv, name=f'class_net_{num_classes}')
    classification = [class_net(feature) for feature in fpn_features]
    classification = layers.Concatenate(axis=1, name='c')(classification)
    # class_net.level = 0
    regression = [box_net(feature) for feature in fpn_features]
    # box_net.level = 0
    regression = layers.Concatenate(axis=1, name='r')(regression)

    model = models.Model(inputs=[image_input], outputs=[classification, regression], name='EfficientDet')

    # apply predicted regression to anchors
    anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    anchors_input = np.expand_dims(anchors, axis=0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    detections = FilterDetections(name='filtered_detections', score_threshold=score_threshold)([boxes, classification])

    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='EfficientDet_p')
    return model, prediction_model


def classification_loss(alpha=0.25, gamma=1.5):
    def focal_loss(y_true, y_pred):
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        indices = tf.where(backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        alpha_factor = backend.ones_like(labels) * alpha
        alpha_factor = tf.where(backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * backend.binary_crossentropy(labels, classification)

        normalizer = tf.where(backend.equal(anchor_state, 1))
        normalizer = backend.cast(backend.shape(normalizer)[0], backend.floatx())
        normalizer = backend.maximum(backend.cast_to_floatx(1.0), normalizer)

        return backend.sum(cls_loss) / normalizer

    return focal_loss


def regression_loss(sigma=3.0):
    sigma_squared = sigma ** 2

    def smooth_l1_loss(y_true, y_pred):
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        indices = tf.where(backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        regression_diff = regression - regression_target
        regression_diff = backend.abs(regression_diff)
        r_loss = tf.where(backend.less(regression_diff, 1.0 / sigma_squared),
                          0.5 * sigma_squared * backend.pow(regression_diff, 2),
                          regression_diff - 0.5 / sigma_squared)

        normalizer = backend.maximum(1, backend.shape(indices)[0])
        normalizer = backend.cast(normalizer, dtype=backend.floatx())
        return backend.sum(r_loss) / normalizer

    return smooth_l1_loss
