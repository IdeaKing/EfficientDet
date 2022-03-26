# Thomas Chia i-Sight Dataset Pipeline

import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
import albumentations as A

from typing import List, Tuple

from utils.anchors import Encoder
from utils import label_utils


class Dataset():
    def __init__(self,
                 file_names: List,
                 dataset_path: str,
                 labels_dict: dict,
                 batch_size: int = 4,
                 shuffle_size: int = 64,
                 images_dir: str = "images",
                 labels_dir: str = "labels",
                 image_dims: Tuple = (512, 512),
                 augment_ds: bool = False,
                 dataset_type: str = "labeled"):
        """ Creates the object detection dataset. """
        self.file_names = file_names
        self.dataset_path = dataset_path
        self.labels_dict = labels_dict
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_dims = image_dims
        self.augment_ds = augment_ds
        self.dataset_type = dataset_type
        self.encoder = Encoder()

    def augment(self, image, label, bbx):
        """For augmenting images and bboxes."""
        # Read and preprocess the image
        image, label, bbx = (image, label.tolist(), bbx.tolist())
        # Augmentation function
        if self.augment_ds:
            transform = A.Compose(
                [A.Flip(p=0.5),
                 A.Rotate(p=0.5),
                 A.RandomBrightnessContrast(p=0.2)],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"]))
        else:
            transform = A.Compose(
                [],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"]))
        aug = transform(
            image=image,
            bboxes=bbx,
            class_labels=label)
        image = np.array(aug["image"], np.float32)
        labels = np.array(aug["class_labels"], np.int32)
        bbx = np.array(aug["bboxes"], np.float32)
        return image, labels, bbx

    def parse_process_voc(self, file_name):
        """Parses the PascalVOC XML Type file."""
        # Reads a voc annotation and returns
        # a list of tuples containing the ground
        # truth boxes and its respective label

        root = ET.parse(file_name).getroot()
        image_size = (int(root.findtext("size/width")),
                      int(root.findtext("size/height")))
        boxes = root.findall("object")
        bbx = []
        labels = []

        for b in boxes:
            bb = b.find("bndbox")
            bb = (int(bb.findtext("xmin")),
                  int(bb.findtext("ymin")),
                  int(bb.findtext("xmax")),
                  int(bb.findtext("ymax")))
            bbx.append(bb)
            labels.append(
                int(self.labels_dict[b.findtext("name")]))
        bbx = tf.stack(bbx)
        # bbx are in relative mode
        bbx = label_utils.to_relative(bbx, image_size)
        # Scale bbx to input image dims
        bbx = label_utils.to_scale(bbx, self.image_dims)
        return np.array(labels), np.array(bbx)

    def parse_process_image(self, file_name):
        image = tf.io.read_file(file_name)
        image = tf.io.decode_jpeg(
            image,
            channels=3)
        image = tf.image.resize(images=image,
                                size=self.image_dims)
        image = np.asarray(image, np.float32)
        return image

    def parse_object_detection(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        image_file_path = os.path.join(self.dataset_path,
                                       self.images_dir,
                                       file_name + ".jpg")
        label_file_path = os.path.join(self.dataset_path,
                                       self.labels_dir,
                                       file_name + ".xml")
        image = self.parse_process_image(
            file_name=image_file_path)
        label, bboxs = self.parse_process_voc(
            file_name=label_file_path)
        image, label, bboxs = self.augment(
            image=image, label=label, bbx=bboxs)
        bboxs = label_utils.to_xywh(bboxs)
        image, label, bboxs = (np.array(image, np.float32),
                               np.array(label, np.int32),
                               np.array(bboxs, np.float32))
        label, bboxs = self.encoder._encode_sample(
            image_shape=self.image_dims,
            gt_boxes=bboxs,
            classes=label)
        return image, label, bboxs

    def __call__(self):
        list_ds = tf.data.Dataset.from_tensor_slices(self.file_names)
        ds = list_ds.map(
            lambda x: tf.numpy_function(
                self.parse_object_detection,
                inp=[x],
                Tout=[tf.float32, tf.float32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            name="object_detection_parser")
        ds = ds.shuffle(self.shuffle_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds


def load_data(dataset_path, file_name="train.txt"):
    """Reads each line of the file."""
    file_names = []
    with open(
        os.path.join(
            dataset_path, file_name)) as reader:
        for line in reader.readlines():
            file_names.append(line.rstrip().split(" ")[0])
    random.shuffle(file_names)
    return file_names
