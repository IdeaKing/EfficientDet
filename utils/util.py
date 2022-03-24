import cv2
import numpy as np
from tensorflow.keras.backend import floatx
from utils.compute_overlap import compute_overlap


class AnchorParameters:
    def __init__(self, sizes=(32, 64, 128, 256, 512),
                 strides=(8, 16, 32, 64, 128),
                 ratios=(1, 0.5, 2),
                 scales=(2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=floatx())
        self.scales = np.array(scales, dtype=floatx())

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


def anchor_targets_bbox(anchors, image, annotations, num_classes, negative_overlap=0.4, positive_overlap=0.5):
    regression_batch = np.zeros((anchors.shape[0], 4 + 1), dtype=np.float32)
    labels_batch = np.zeros((anchors.shape[0], num_classes + 1), dtype=np.float32)

    if annotations['bboxes'].shape[0]:
        positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors,
                                                                                        annotations['bboxes'],
                                                                                        negative_overlap,
                                                                                        positive_overlap)
        labels_batch[ignore_indices, -1] = -1
        labels_batch[positive_indices, -1] = 1

        regression_batch[ignore_indices, -1] = -1
        regression_batch[positive_indices, -1] = 1

        labels_batch[positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

        regression_batch[:, :4] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

    if image.shape:
        anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
        indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

        labels_batch[indices, -1] = -1
        regression_batch[indices, -1] = -1

    return labels_batch, regression_batch


def compute_gt_annotations(anchors, annotations, negative_overlap=0.4, positive_overlap=0.5):
    # (N, K)
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # (N, )
    argmax_overlaps_indices = np.argmax(overlaps, axis=1)
    # (N, )
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_indices]

    # assign "dont care" labels
    # (N, )
    positive_indices = max_overlaps >= positive_overlap

    # in case of there are gt boxes has no matched positive anchors
    nonzero_indices = np.nonzero(overlaps == np.max(overlaps, axis=0))
    positive_indices[nonzero_indices[0]] = 1

    # (N, )
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_indices


def shapes_callback(image_shape, pyramid_levels):
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(image_shape, anchor_params=None):
    pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters()

    feature_map_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4), dtype=np.float32)
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=anchor_params.sizes[idx],
                                   ratios=anchor_params.ratios,
                                   scales=anchor_params.scales)
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)


def shift(feature_map_shape, stride, anchors):
    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = AnchorParameters().ratios

    if scales is None:
        scales = AnchorParameters().scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, scale_factors=None):
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + wa / 2.
    cya = anchors[:, 1] + ha / 2.

    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    cx = gt_boxes[:, 0] + w / 2.
    cy = gt_boxes[:, 1] + h / 2.
    # Avoid NaN in division and log below.
    ha += 1e-7
    wa += 1e-7
    h += 1e-7
    w += 1e-7
    tx = (cx - cxa) / wa
    ty = (cy - cya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    if scale_factors:
        ty /= scale_factors[0]
        tx /= scale_factors[1]
        th /= scale_factors[2]
        tw /= scale_factors[3]
    targets = np.stack([ty, tx, th, tw], axis=1)
    return targets


def brightness(image, prob=0.5, min_factor=0.7, max_factor=1.):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image

    factor = np.random.uniform(min_factor, max_factor)
    table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image


class VisualEffect:
    def __init__(self):
        pass

    def __call__(self, image):
        return brightness(image)
