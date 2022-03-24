from os.path import join
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
from six import raise_from

from utils import config
from utils.util import anchors_for_shape, anchor_targets_bbox, AnchorParameters, VisualEffect


def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


def name_to_label(name):
    return config.classes[name]


def load_image(f_name):
    path = join(config.data_dir, config.image_dir, f_name + '.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def parse_annotation(element):
    truncated = find_node(element, 'truncated', parse=int)
    difficult = find_node(element, 'difficult', parse=int)

    class_name = find_node(element, 'name').text
    if class_name not in config.classes:
        raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(config.classes.keys())))

    box = np.zeros((4,))
    label = name_to_label(class_name)

    bndbox = find_node(element, 'bndbox')
    box[0] = find_node(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
    box[1] = find_node(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
    box[2] = find_node(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
    box[3] = find_node(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

    return truncated, difficult, box, label


def parse_annotations(xml_root):
    annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}
    for i, element in enumerate(xml_root.iter('object')):
        try:
            truncated, difficult, box, label = parse_annotation(element)
        except ValueError as e:
            raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

        if truncated and False:
            continue
        if difficult:
            continue

        annotations['bboxes'] = np.concatenate([annotations['bboxes'], [box]])
        annotations['labels'] = np.concatenate([annotations['labels'], [label]])

    return annotations


def load_label(f_name):
    try:
        tree = parse_fn(join(config.data_dir, config.label_dir, f_name + '.xml'))
        return parse_annotations(tree.getroot())
    except ParseError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)
    except ValueError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)


def filter_annotations(image, label):
    # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
    invalid_indices = np.where((label['bboxes'][:, 2] <= label['bboxes'][:, 0]) |
                               (label['bboxes'][:, 3] <= label['bboxes'][:, 1]) |
                               (label['bboxes'][:, 0] < 0) |
                               (label['bboxes'][:, 1] < 0) |
                               (label['bboxes'][:, 2] <= 0) |
                               (label['bboxes'][:, 3] <= 0) |
                               (label['bboxes'][:, 2] > image.shape[1]) |
                               (label['bboxes'][:, 3] > image.shape[0]))[0]

    if len(invalid_indices):
        for k in label.keys():
            label[k] = np.delete(label[k], invalid_indices, axis=0)
    return image, label


def preprocess_image(image):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = config.image_size / image_height
        resized_height = config.image_size
        resized_width = int(image_width * scale)
    else:
        scale = config.image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = config.image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = config.image_size - resized_height
    pad_w = config.image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
    return image, scale


def clip_transformed_annotations(image, annotations):
    image_height = image.shape[0]
    image_width = image.shape[1]
    # x1
    annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
    # y1
    annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
    # x2
    annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
    # y2
    annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
    # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
    small_indices = np.where((annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 3) |
                             (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 3))[0]

    if len(small_indices):
        for k in annotations.keys():
            annotations[k] = np.delete(annotations[k], small_indices, axis=0)
    return image, annotations


def generator_fn(f_names):
    def generator():
        index = 0
        while 1:
            f_name = f_names[index]
            image = load_image(f_name)
            label = load_label(f_name)

            image, label = filter_annotations(image, label)

            image = VisualEffect()(image)

            image, scale = preprocess_image(image)
            label['bboxes'] *= scale

            image, label = clip_transformed_annotations(image, label)

            anchors = anchors_for_shape((config.image_size, config.image_size), AnchorParameters())
            label = anchor_targets_bbox(anchors, image, label, len(config.classes))
            index += 1
            if index == len(f_names):
                index = 0
                np.random.shuffle(f_names)
            yield image, label[0], label[1]

    return generator
