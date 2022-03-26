"""
Run the inference on the EfficientNet model.

Thomas Chia
"""

import os
import argparse
import shutil
import tensorflow as tf

from tqdm import tqdm

from utils.postprocess import FilterDetections
from utils.visualize import draw_boxes
from utils.file_reader import parse_label_file


def preprocess_image(image_path: str, 
                     image_dims: tuple) -> tf.Tensor:
    """Preprocesses an image.
        
    Parameters:
        image_path: Path to the image
        image_dims: The dimensions to resize the image to
    Returns:
        A preprocessed image with range [0, 255]
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image)
    image = tf.image.resize(images=image,
                            size=image_dims,
                            method="bilinear")
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    # Image is on scale [0-255]
    return image


def test(image_path: str, 
         image_dir: str, 
         save_dir: str, 
         model: tf.keras.models.Model,
         image_dims: tuple, 
         label_dict: dict, 
         score_threshold: float, 
         iou_threshold: float) -> None:
    """Preprocesses, Tests, and Postprocesses.
    
    Parameters:
        image_path: Path to image to test
        image_dir: Path to directory containing images
        save_dir: Path to directory to save images
        model: Model to test
        image_dims: Dimensions of image
        label_dict: Dictionary mapping labels to names
        score_threshold: Threshold for score
        iou_threshold: Threshold for iou
    Returns:
        None
    """
    image = preprocess_image(
        os.path.join(image_dir, image_path), image_dims)

    pred_cls, pred_box = model(image, training=False)
    labels, bboxes, scores = FilterDetections(
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        image_dims=image_dims)(
            labels=pred_cls,
            bboxes=pred_box)

    labels = [list(label_dict.keys())[int(l)]
              for l in labels[0]]
    bboxes = bboxes[0]
    scores = scores[0]

    image = draw_boxes(
        image=tf.squeeze(image, axis=0),
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        labels_dict=label_dict)

    image.save(os.path.join(save_dir, image_path))


if __name__ == "__main__":
    # Supress warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run EfficientDet Tests",
        prog="EfficientDet")
    parser.add_argument("--testing-image-dir",
                        type=str,
                        default="datasets/data/VOC2012/test",
                        help="Path to testing images directory.")
    parser.add_argument("--save-image-dir",
                        type=str,
                        default="data/datasets/tests/VOC2012/test",
                        help="Path to testing images directory.")
    parser.add_argument("--model-dir",
                        type=str,
                        default="model-exported",
                        help="Path to testing model directory.")
    parser.add_argument("--image-dims",
                        type=tuple,
                        default=(512, 512),
                        help="Size of the input image.")
    parser.add_argument("--labels-file",
                        type=str,
                        default="datasets/data/VOC2012/labels.txt",
                        help="Path to labels file.")
    parser.add_argument("--score-threshold",
                        type=float,
                        default=0.35,
                        help="Score threshold for NMS.")
    parser.add_argument("--iou-threshold",
                        type=float,
                        default=0.5,
                        help="IOU threshold for NMS.")
    args = parser.parse_args()

    label_dict = parse_label_file(
        path_to_label_file=args.labels_file)

    print(f"Loading model from {args.model_dir}")
    model = tf.keras.models.load_model(args.model_dir)

    if os.path.exists(args.save_image_dir) == False:
        print(f"Making the save directory {args.save_image_dir}")
        os.mkdir(args.save_image_dir)
    else:
        input(f"Press Enter to delete {args.save_image_dir} and continue.")
        shutil.rmtree(args.save_image_dir)
        os.mkdir(args.save_image_dir)

    print(f"Testing the model on the images in {args.testing_image_dir}")
    for image_path in tqdm(os.listdir(args.testing_image_dir)):
        # Test the model on the image
        test(image_path=image_path,
             image_dir=args.testing_image_dir,
             save_dir=args.save_image_dir,
             model=model,
             image_dims=args.image_dims,
             label_dict=label_dict,
             score_threshold=args.score_threshold,
             iou_threshold=args.iou_threshold)
