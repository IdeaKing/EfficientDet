"""Implementation of utility functions."""

import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def to_xywh(bbox):
    """Convert [x_min, y_min, x_max, y_max] to [x, y, width, height]."""
    return tf.concat(
        [(bbox[..., :2] + bbox[..., 2:]) / 2.0, (bbox[..., 2:] - bbox[..., :2])], axis=-1
    )


@tf.autograph.experimental.do_not_convert
def to_tf_format(boxes: tf.Tensor) -> tf.Tensor:
    """
    Convert xmin, ymin, xmax, ymax boxes to ymin, xmin, ymax, xmax
    and viceversa
    """
    x1, y1, x2, y2 = tf.split(boxes[..., :], 4, axis=-1)
    return tf.concat([y1, x1, y2, x2], axis=-1)


@tf.autograph.experimental.do_not_convert
def to_norm_format(boxes: tf.Tensor) -> tf.Tensor:
    """
    Convert ymin, xmin, ymax, xmax boxes to xmin, ymin, xmax, ymax
    and viceversa
    """
    y1, x1, y2, x2 = tf.split(boxes[..., :], 4, axis=-1)
    return tf.concat([x1, y1, x2, y2], axis=-1)


@tf.autograph.experimental.do_not_convert
def to_corners(bbox):
    """Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    return tf.concat(
        [bbox[..., :2] - bbox[..., 2:] / 2.0, bbox[..., :2] + bbox[..., 2:] / 2.0], axis=-1
    )


@tf.autograph.experimental.do_not_convert
def to_relative(bbox, image_dims):
    """Convert pixel wise ground truth labels to relative ground truth."""
    (width, height) = image_dims
    x1, y1, x2, y2 = tf.split(bbox, 4, axis=1)
    x1 /= (width)
    x2 /= (width)
    y1 /= (height)
    y2 /= (height)
    return tf.concat([x1, y1, x2, y2], axis=1)


@tf.autograph.experimental.do_not_convert
def match_anchors(boxes, anchor_boxes):
    box_variance = tf.cast(
        [0.1, 0.1, 0.2, 0.2], tf.float32)
    boxes = boxes * box_variance
    boxes = tf.concat(
        [boxes[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
         tf.exp(boxes[..., 2:]) * anchor_boxes[..., 2:]],
        axis=-1)
    boxes = to_corners(boxes)
    return boxes


@tf.autograph.experimental.do_not_convert
def to_scale(bbox, image_dims):
    """Convert pixel wise ground truth labels to relative ground truth."""
    (width, height) = image_dims
    x1, y1, x2, y2 = tf.split(bbox, 4, axis=1)
    x1 *= (width)
    x2 *= (width)
    y1 *= (height)
    y2 *= (height)
    return tf.concat([x1, y1, x2, y2], axis=1)


@tf.autograph.experimental.do_not_convert
def compute_iou(boxes_1, boxes_2):
    """Compute intersection over union.
    Args:
        boxes_1: a tensor with shape (N, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].
        boxes_2: a tensor with shape (M, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].
    Returns:
        IOU matrix with shape (N, M).
    """

    boxes_1_corners = to_corners(boxes_1)
    boxes_2_corners = to_corners(boxes_2)

    left_upper = tf.maximum(
        boxes_1_corners[..., None, :2], boxes_2_corners[..., :2])
    right_lower = tf.minimum(
        boxes_1_corners[..., None, 2:], boxes_2_corners[..., 2:])
    diff = tf.maximum(0.0, right_lower - left_upper)
    intersection = diff[..., 0] * diff[..., 1]

    boxes_1_area = boxes_1[..., 2] * boxes_1[..., 3]
    boxes_2_area = boxes_2[..., 2] * boxes_2[..., 3]
    union = boxes_1_area[..., None] + boxes_2_area - intersection

    iou = intersection / union
    return tf.clip_by_value(iou, 0.0, 1.0)
