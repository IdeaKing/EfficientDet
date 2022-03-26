import tensorflow as tf

from typing import Tuple

from utils import anchors, label_utils
from utils import label_utils


class FilterDetections:
    def __init__(self,
                 score_threshold: float = 0.3,
                 image_dims: Tuple[int, int] = (512, 512),
                 max_boxes: int = 150,
                 max_size: int = 100,
                 iou_threshold: int = 0.8):

        self.score_threshold = score_threshold
        self.image_dims = image_dims
        self.max_boxes = max_boxes
        self.max_size = max_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.anchors = anchors.Anchors().get_anchors(
            image_height=image_dims[0],
            image_width=image_dims[1])

    def __call__(self,
                 labels: tf.Tensor,
                 bboxes: tf.Tensor):

        labels = tf.nn.sigmoid(labels)
        # bboxes: (batch_size, x_center, y_center, width, height)
        bboxes = label_utils.match_anchors(
            boxes=bboxes,
            anchor_boxes=self.anchors)
        # bboxes: (batch_size, x_min, y_min, x_max, y_max)
        tf_bboxes = label_utils.to_tf_format(bboxes)
        # bboxes: (batch_size, y_min, x_min, y_max, x_max)
        nms = tf.image.combined_non_max_suppression(
            tf.expand_dims(tf_bboxes, axis=2),
            labels,
            max_output_size_per_class=self.max_boxes,
            max_total_size=self.max_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            clip_boxes=False,
            name="Combined-NMS")

        labels = nms.nmsed_classes
        bboxes = label_utils.to_norm_format(nms.nmsed_boxes)
        # bboxes: (batch_size, x_min, y_min, x_max, y_max)
        scores = nms.nmsed_scores

        return labels, bboxes, scores
