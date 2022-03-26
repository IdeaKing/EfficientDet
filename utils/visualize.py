"""
Display functions.

Thomas Chia
"""

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image


def draw_boxes(image: tf.Tensor,
               bboxes: list,
               labels: list,
               scores: list,
               labels_dict: dict) -> Image:
    """ Draw a set of boxes formatted as [x1, y1, x2, y2] to image.

    Parameters:
        images: A tensor of shape [batch_size, height, width, channels]
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

    if image.dtype == "float32" or image.dtype == "float64":
        image = (image).astype("uint8")
    elif image.dtype != "uint8":
        print(image.dtype)
        raise ValueError("Image dtype not supported")

    # Generate one color per class
    n_colors = len(labels_dict)
    color_pallete = [tuple(np.random.choice(range(255), size=3))
                     for color in range(n_colors)]

    # Generates the images from arrays
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    # Loop through the boxes and draw them on image
    for boxes, label, score in zip(bboxes, labels, scores):
        if score < 0.01:
            # Skip low confidence boxes
            continue
        text = str(f"{label} {round(float(score), 3)}")
        x1, y1, x2, y2 = boxes
        c = color_pallete[labels_dict[label] % n_colors]
        draw.text([x1 - 5, y1 + 10], text)
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
