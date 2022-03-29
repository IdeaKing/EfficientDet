"""
Display functions.

Thomas Chia
"""

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image, ImageFont


def draw_boxes(image: tf.Tensor,
               original_shape: tuple,
               resized_shape: tuple,
               bboxes: list,
               labels: list,
               scores: list,
               labels_dict: dict) -> Image:
    """ Draw a set of boxes formatted as [x1, y1, x2, y2] to image.

    Parameters:
        images: A tensor of shape [batch_size, height, width, channels]
        original_shape: The original shape of the image (h, w)
        resized_shape: Shape of the image after resizing into the model (w, h)
        bboxes: A list of bounding boxes formatted as [x1, y1, x2, y2]
        labels: A list of labels corresponding to the bounding boxes
        scores: A list of scores corresponding to the bounding boxes
        labels_dict: A dictionary mapping labels to names
    Returns:
        A PIL image of the boxes drawn on the image
    """

    if isinstance(image, Image.Image):
        image = image
    elif isinstance(image, tf.Tensor):
        image = np.array(image)
    else: 
        raise ValueError("Image type is not allowed.")

    if image.dtype == "float32" or image.dtype == "float64":
        image = (image).astype("uint8")
    elif image.dtype != "uint8":
        print(image.dtype)
        raise ValueError("Image dtype not supported")

    # Generate one color per class
    n_colors = len(labels_dict)
    color_pallete = [tuple(np.random.choice(range(155), size=3) + 100)
                     for color in range(n_colors)]

    # Generates the images from arrays
    image = Image.fromarray(image)
    # Resize image
    image = image.resize(original_shape)
    # Drawing functions and settings
    draw = ImageDraw.Draw(image)

    # Loop through the boxes and draw them on image
    for boxes, label, score in zip(bboxes, labels, scores):
        if score < 0.01:
            # Skip low confidence boxes
            continue
        text = str(f"{label} {round(float(score), 3)}")
        x1 = float((boxes[0]/resized_shape[0]) * float(original_shape[0]))
        x2 = float((boxes[2]/resized_shape[0]) * float(original_shape[0]))
        y1 = float((boxes[1]/resized_shape[1]) * float(original_shape[1]))
        y2 = float((boxes[3]/resized_shape[1]) * float(original_shape[1]))
        boxes = (x1, y1, x2, y2)

        c = color_pallete[labels_dict[label] % n_colors]
        draw.text([x1 + 5, y1 + 5], text)
        draw.rectangle(boxes, outline=c, width=2)

    return image


def printProgressBar(
        step,
        total,
        loss_vals):
    """Training Progress Bar"""
    decimals = 1
    length = 20
    fill = "="  # "█"
    printEnd = "\r"

    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (step / float(total)))
    filledLength = int(length * step // total)
    bar = fill * filledLength + "-" * (length - filledLength)

    print(f"\r Step: {step}/{total} |{bar}| {percent}%" +
          " ".join(f"loss-{i} {round(loss, 5)}" for i,
                   loss in enumerate(loss_vals)),
          end=printEnd)

    # Print New Line on Complete
    if iter == total:
        print()
