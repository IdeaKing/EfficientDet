import os
import pickle
from os.path import join, exists, basename
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nets.nn import build_model
from utils import config
from utils.data_loader import find_node


def preprocess_image(image, image_size):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale


def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def draw_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = list(map(int, box))
        cv2.putText(image, 
                    '{}'.format(label), 
                    (xmin, ymin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)


def main():
    score_threshold = 0.2
    if not exists(join('results', f'D{config.phi}')):
        os.makedirs(join('results', f'D{config.phi}'))
    train_model, inference_model = build_model(phi=config.phi,
                                               num_classes=len(config.classes),
                                               score_threshold=score_threshold)
    train_model.load_weights(config.weight_path)
    train_model.save_weights(config.weight_path)
    inference_model.load_weights(config.weight_path, by_name=True)
    f_names = []
    with open(join(config.data_dir, 'val.txt')) as reader:
        for line in reader.readlines():
            f_names.append(line.rstrip().split(' ')[0])
    result_dict = {}
    for f_name in tqdm.tqdm(f_names):
        image_path = join(config.data_dir, config.image_dir, f_name + '.jpg')
        label_path = join(config.data_dir, config.label_dir, f_name + '.xml')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        src_image = image.copy()
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=config.image_size)
        boxes, scores, labels = inference_model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        indices = np.where(scores[:] > score_threshold)[0]

        boxes = boxes[indices]
        labels = [list(config.classes.keys())[int(l)]
                  for l in labels]
        draw_boxes(src_image, boxes, labels)
        pred_boxes_np = []
        for pred_box in boxes:
            x_min, y_min, x_max, y_max = pred_box
            pred_boxes_np.append([x_min, y_min, x_max, y_max])
        true_boxes = []
        for element in parse_fn(label_path).getroot().iter('object'):
            box = find_node(element, 'bndbox')
            x_min = find_node(box, 'xmin', 'bndbox.xmin', parse=float) - 1
            y_min = find_node(box, 'ymin', 'bndbox.ymin', parse=float) - 1
            x_max = find_node(box, 'xmax', 'bndbox.xmax', parse=float) - 1
            y_max = find_node(box, 'ymax', 'bndbox.ymax', parse=float) - 1
            true_boxes.append([x_min, y_min, x_max, y_max])
        result = {'detection_boxes': pred_boxes_np,
                  'groundtruth_boxes': true_boxes,
                  'confidence': scores}
        result_dict[f'{f_name}.jpg'] = result
        cv2.imwrite(join('results', f'D{config.phi}', basename(image_path)), src_image[:, :, ::-1])
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', src_image)
        # cv2.waitKey(0)
    with open(join('results', f'D{config.phi}', 'd4.pickle'), 'wb') as writer:
        pickle.dump(result_dict, writer)


if __name__ == '__main__':
    main()
