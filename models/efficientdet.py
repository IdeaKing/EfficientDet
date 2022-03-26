"""Implementation of EffDets."""

import tensorflow as tf
from .layers import BiFPN, ClassDetector, BoxRegressor
from .backbones import get_backbone

from typing import Tuple

class EfficientDet(tf.keras.Model):
    """EfficientDet model."""

    def __init__(self,
                 channels=64,
                 num_classes=80,
                 num_anchors=9,
                 bifpn_depth=3,
                 bifpn_kernel_size=3,
                 bifpn_depth_multiplier=1,
                 bifpn_pooling_strategy="avg",
                 heads_depth=3,
                 class_kernel_size=3,
                 class_depth_multiplier=1,
                 box_kernel_size=3,
                 box_depth_multiplier=1,
                 backbone_name="efficientnet_b0",
                 name="efficientdet_d0"):
        """Initialize EffDet. Default args refers to EfficientDet D0.
        Args:
            channels: an integer representing number of units inside each fusing
                node and convolution layer of BiFPN and head models.
            num_classes: an integer representing number of classes to predict.
            num_anchors: an integer representing number of anchor boxes.
            bifpn_depth: an integer representing number of BiFPN layers.
            bifpn_kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window for BiFPN layers.
            bifpn_depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in BiFPN nodes.
            bifpn_pooling_strategy: a string representing pooling strategy in BiFPN
                layers. "avg" or "max". Otherwise the max pooling will be selected.
            heads_depth: an integer representing number of separable convolutions
                before final convolution in head models.
            class_kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window for
                classifier model.
            class_depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in classifier model.
            box_kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window for
                regression model.
            box_depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in regression model.
            name: a string representing model name.
        """
        super().__init__(name=name)
        self.num_classes=num_classes

        self.backbone = get_backbone(backbone_name)
        self.backbone.trainable = False

        self.BiFPN = BiFPN(channels=channels,
                           depth=bifpn_depth,
                           kernel_size=bifpn_kernel_size,
                           depth_multiplier=bifpn_depth_multiplier,
                           pooling_strategy=bifpn_pooling_strategy)
        self.class_det = ClassDetector(channels=channels,
                                       num_classes=num_classes,
                                       num_anchors=num_anchors,
                                       depth=heads_depth,
                                       kernel_size=class_kernel_size,
                                       depth_multiplier=class_depth_multiplier)
        self.box_reg = BoxRegressor(channels=channels,
                                    num_anchors=num_anchors,
                                    depth=heads_depth,
                                    kernel_size=box_kernel_size,
                                    depth_multiplier=box_depth_multiplier)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        features = self.backbone(inputs)
        features.append(tf.keras.layers.AveragePooling2D()(features[-1]))
        features.append(tf.keras.layers.AveragePooling2D()(features[-1]))

        fpn_features = self.BiFPN(features, training=training)

        classes = list()
        boxes = list()
        for feature in fpn_features:
            classes.append(tf.reshape(self.class_det(feature, training=training), [batch_size, -1, self.num_classes]))
            boxes.append(tf.reshape(self.box_reg(feature, training=training), [batch_size, -1, 4]))

        classes = tf.concat(classes, axis=1)
        boxes = tf.concat(boxes, axis=1)

        return classes, boxes


def get_efficientdet(name: str = "efficientdet_d0",
                     input_shape: Tuple[int, int] = (512, 512),
                     num_classes: int = 20,
                     num_anchors: int = 9) -> tf.keras.models.Model:
    models = {
        "efficientdet_d0": (64, 3, 3, "efficientnet_b0"),
        "efficientdet_d1": (88, 4, 3, "efficientnet_b1"),
        "efficientdet_d2": (112, 5, 3, "efficientnet_b2"),
        "efficientdet_d3": (160, 6, 4, "efficientnet_b3"),
        "efficientdet_d4": (224, 7, 4, "efficientnet_b4"),
        "efficientdet_d5": (288, 7, 4, "efficientnet_b5"),
        "efficientdet_d6": (384, 8, 5, "efficientnet_b6"),
        "efficientdet_d7": (384, 8, 5, "efficientnet_b7"),
    }
    input_layer = tf.keras.layers.Input((*input_shape, 3))
    classes, boxes = EfficientDet(channels=models[name][0],
                                  num_classes=num_classes,
                                  num_anchors=num_anchors,
                                  bifpn_depth=models[name][1],
                                  heads_depth=models[name][2],
                                  name=name)(input_layer)
    return tf.keras.models.Model(inputs=[input_layer],
                                 outputs=[classes, boxes])