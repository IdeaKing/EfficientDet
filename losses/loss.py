import numpy as np
import tensorflow as tf

from utils import label_utils
from losses import iou_utils


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss implementations."""

    def __init__(self,
                 alpha=0.25,
                 gamma=1.5,
                 label_smoothing=0.1,
                 name="focal_loss"):
        """Initialize parameters for Focal loss.
        FL = - alpha_t * (1 - p_t) ** gamma * log(p_t)
        This implementation also includes label smoothing for preventing overconfidence.
        """
        super().__init__(name=name, reduction="none")
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """Calculate Focal loss.
        Args:
            y_true: a tensor of ground truth values with
                shape (batch_size, num_anchor_boxes, num_classes).
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).
        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        prob = tf.sigmoid(y_pred)
        pt = y_true * prob + (1 - y_true) * (1 - prob)
        at = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        y_true = y_true * (1.0 - self.label_smoothing) + \
            0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)

        loss = at * (1.0 - pt)**self.gamma * ce
        return tf.reduce_sum(loss, axis=-1)


class BoxLoss(tf.keras.losses.Loss):
    """Huber loss implementation."""

    def __init__(self,
                 delta=1.0,
                 name="box_loss"):
        super().__init__(name=name, reduction="none")
        self.delta = delta

    def call(self, y_true, y_pred):
        """Calculate Huber loss.
        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 4).
            y_pred: a tensor of predicted values with shape (batch_size, num_anchor_boxes, 4).
        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        loss = tf.abs(y_true - y_pred)
        l1 = self.delta * (loss - 0.5 * self.delta)
        l2 = 0.5 * loss ** 2
        box_loss = tf.where(tf.less(loss, self.delta), l2, l1)
        return tf.reduce_sum(box_loss, axis=-1)


class EffDetLoss(tf.keras.losses.Loss):
    """Composition of Focal and Huber losses."""

    def __init__(self,
                 num_classes,
                 alpha=0.25,
                 gamma=1.5,
                 label_smoothing=0.1,
                 delta=1.0,
                 include_iou=None,
                 name="effdet_loss"):
        """Initialize Focal and Huber loss.
        Args:
            num_classes: an integer number representing number of
                all possible classes in training dataset.
            alpha: a float number for Focal loss formula.
            gamma: a float number for Focal loss formula.
            label_smoothing: a float number of label smoothing intensity.
            delta: a float number representing a threshold in Huber loss
                for choosing between linear and cubic loss.
            include_iou: either None or "ciou", "diou", "iou", "giou"
        """
        super().__init__(name=name)
        self.class_loss = FocalLoss(
            alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.box_loss = BoxLoss(delta=delta)
        self.num_classes = num_classes
        self.include_iou = include_iou

    # @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """Calculate Focal and Huber losses for every anchor box.
        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 5)
                representing anchor box correction and class label.
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).
        Returns:
            loss: a float loss value.
        """
        box_preds = tf.cast(y_pred[1], dtype=tf.float32)
        cls_preds = tf.cast(y_pred[0], dtype=tf.float32)

        box_labels = y_true[1]
        cls_labels = tf.squeeze(
            tf.one_hot(
                tf.cast(y_true[0], dtype=tf.int32),
                depth=self.num_classes,
                dtype=tf.float32),
            axis=2)

        positive_mask = tf.cast(tf.greater(y_true[0], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[0], -2.0), dtype=tf.float32)

        clf_loss = self.class_loss(cls_labels, cls_preds)
        box_loss = self.box_loss(box_labels, box_preds)

        positive_mask = tf.squeeze(positive_mask, 2)
        ignore_mask = tf.squeeze(ignore_mask, 2)

        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(
            tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(
            tf.reduce_sum(box_loss, axis=-1), normalizer)

        if self.include_iou is None:
            loss = clf_loss + box_loss
        else:
            box_preds = label_utils.to_tf_format(box_preds)
            iou = iou_utils.iou_loss(
                pred_boxes=box_preds,
                target_boxes=box_labels,
                iou_type=self.include_iou)
            loss = clf_loss + box_loss + iou
        return loss


class UDA(tf.keras.losses.Loss):
    """UDA Loss Function."""

    def __init__(
            self,
            batch_size: int,
            unlabeled_batch_size: int,
            num_classes: int,
            loss_func: tf.keras.losses.Loss,
            training_type: str = "object_detection"):
        """Unsupervised Data Augmentation for Consistency Training

        Parameters:
            batch_size (int): The batch size
            unlabeled_batch_size (int): The unlabeled batch size
            num_classes (int): The number of classes
            loss_func (tf.keras.losses.loss): The loss function to 
                apply labeled data
            training_type (str): Either "object_detection", "segmentation",
                or "classification"
        """
        super(UDA, self).__init__()
        self.batch_size = batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.num_classes = num_classes
        self.loss_func = loss_func

        if training_type == "object_detection":
            self.convert_to_labels = pseudo_labels.PseudoLabelObjectDetection()
            self.consistency_loss = tf.keras.losses.KLDivergence()

    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        labels = y_true
        masks = {}
        logits = {}
        loss = {}
        # Splits the predictions for labeled, and unlabeled
        logits["l"], logits["u_ori"], logits["u_aug"] = tf.split(
            y_pred,
            [self.configs.batch_size,
             self.configs.unlabeled_batch_size,
             self.configs.unlabeled_batch_size],
            axis=0)
        # Step 1: Loss for Labeled Values
        loss["l"] = self.loss_func(
            y_true=labels["l"],
            y_pred=logits["l"])
        # Step 2: Loss for unlabeled values
        labels["u_ori"] = self.convert_to_labels(
            logits["u_ori"])  # Applies NMS, anchors
        # Consistency loss between unlabeled values
        if self.training_type == "object_detection":
            unlabeled_cls_loss = self.consistency_loss(
                y_true=labels["u_ori"][0],
                y_pred=logits["u_aug"][0])
            unlabeled_obd_loss = self.consistency_loss(
                y_true=labels["u_ori"][1],
                y_pred=logits["u_aug"][1])
            loss["u"] = tf.reduce_mean(
                [unlabeled_cls_loss, unlabeled_obd_loss])
        return logits, labels, masks, loss
