# Tensorflow2.0 EfficientDet

A pure WORKING Tensorflow2.0 implementation of EfficientDet for object detection. There are too many non-working versions of EfficientDet available. This attempt uses pure tf2.

## Table of Contents

* [Installation](#installation)
* [Preparing Dataset](#preparing-the-dataset)
* [Training](#training)
* [Testing](#testing)
* [Pretrained Weights](#pretrained-weights)
* [References](#references)
* [License](#license)


## Sample Detections

| Dog | People |
| --- | --- |
|![Dog](https://github.com/IdeaKing/EfficientDet/blob/main/docs/2010_000183.jpg) | ![People](https://github.com/IdeaKing/EfficientDet/blob/main/docs/2010_000358.jpg) |

| Bird | Plane |
| --- | --- |
|![Bird](https://github.com/IdeaKing/EfficientDet/blob/main/docs/2010_000184.jpg) | ![Plane](https://github.com/IdeaKing/EfficientDet/blob/main/docs/2010_000392.jpg) |

## Installation

```python
git clone https://github.com/IdeaKing/EfficientDet.git
cd efficientdet
python -m venv venv 
python -m pip install --upgrade pip
venv\Scripts\activate
pip install -r requirements.txt
```

## Preparing the Dataset

You can train your own dataset or use the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

All labels must be in xml Pascal VOC format.

### Training on your own dataset

Use [LabelMe](https://github.com/wkentaro/labelme) to label and create your own dataset. Make sure to save in PascalVOC format.

```python
pip install labelme
labelme
```

Then copy the file names of the images and respesctive labels to the `train.txt` and `val.txt` files. The file names should not include the file type.

### Training on PascalVOC

Download the PascalVOC 2012 dataset from [host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

Then copy the `VOCdevkit/VOC2012` folder to `/data/datasets/VOC2012`. Lastly, copy the train.txt and val.txt files from `data/datasets/VOC2012/ImageSets` to `data/datasets/VOC2012`.

### Creating the labels file

Then create a file with the labels/classes that will be trained on.

```none
cat
dog
cow
person
```

### Dataset Directory Structure

The structure of the directory is as follows:

```files
+-- data
|   +-- datasets
|       +-- VOC2012
|           +-- Annotations
|           +-- ImageSets
|           +-- JPEGImages
|           +-- SegmentationClass
|           +-- SegmentationObject
|           +-- ImageSets
|           +-- train.txt
|           +-- val.txt
|           +-- labels.txt
|       +-- MyDataset
|           +-- Labels
|               +-- image-0001.xml
|               +-- image-0002.xml
|               +-- ...
|           +-- Images
|               +-- image-0001.jpg
|               +-- image-0002.jpg
|               +-- ...
|           +-- Test
|               +-- image-0001.jpg
|               +-- image-0002.jpg
|               +-- ...
|           +-- train.txt
|           +-- val.txt
|           +-- labels.txt
```

## Training

Run the following command to train the model:

```cmd
python main.py --dataset-path <path_to_dataset> \
            --training-dir <path_to_training_dir> \
            --model efficientdet-d0 \
            --debug False \
            --precision float32 \
            --batch-size 8 \
            --epochs 100 \
            --learning-rate 0.0005 \
            --optimizer ADAM \
            --dataset-files train.txt \
            --labels-file labels.txt \
            --images-dir JPEGImages|MyDataset \
            --labels-dir Annotations|Labels \
            --augment-ds True 
```

### All commands

```cmd
(venv) D:\EfficientDet>python main.py -h
usage: EfficientDet [-h] [--dataset-path DATASET_PATH] [--training-dir TRAINING_DIR]
                    [--model MODEL] [--debug DEBUG] [--precision PRECISION]
                    [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--optimizer OPTIMIZER]
                    [--learning-rate LEARNING_RATE]
                    [--optimizer-momentum OPTIMIZER_MOMENTUM]
                    [--from-checkpoint FROM_CHECKPOINT] [--dataset-files DATASET_FILES]
                    [--labels-file LABELS_FILE] [--images-dir IMAGES_DIR]
                    [--labels-dir LABELS_DIR] [--shuffle-size SHUFFLE_SIZE]
                    [--image-dims IMAGE_DIMS] [--augment-ds AUGMENT_DS]
                    [--print-loss PRINT_LOSS] [--log-every-step LOG_EVERY_STEP]
                    [--max-checkpoints MAX_CHECKPOINTS]
                    [--save-model-frequency SAVE_MODEL_FREQUENCY]
                    [--checkpoint-frequency CHECKPOINT_FREQUENCY]

Run EfficientDet Training or Tests

optional arguments:
-h, --help            show this help message and exit
--dataset-path DATASET_PATH
                        Path to dataset
--training-dir TRAINING_DIR
                        Path to the training directory
--model MODEL         Model name, can be efficientdet_d[0 - 7] (object detection)
--debug DEBUG         Use debugging mode or not, NOTE: SEVERELY REDUCES PPERFORMANCE
--precision PRECISION
                        The precision type, can either be mixed_float16 or float32
--batch-size BATCH_SIZE
                        Batch size for training
--epochs EPOCHS       The number of epochs to train the model(s)
--optimizer OPTIMIZER
                        The optimizer to train the model on, can be: SGD or ADAM
--learning-rate LEARNING_RATE
                        The initial learning rate for the optimizer
--optimizer-momentum OPTIMIZER_MOMENTUM
                        The momentum for the optimizer
--from-checkpoint FROM_CHECKPOINT
                        Continue training from checkpoint.
--dataset-files DATASET_FILES
                        Either filename of labeled_train.txt
--labels-file LABELS_FILE
                        Filename of the labels file.
--images-dir IMAGES_DIR
                        Folder name that holds images.
--labels-dir LABELS_DIR
                        Folder name that holds labels.
--shuffle-size SHUFFLE_SIZE
                        Shuffle the dataset steps, keep in powers of 2
--image-dims IMAGE_DIMS
                        Images dims to fit in the model
--augment-ds AUGMENT_DS
                        If augmentation is needed
--print-loss PRINT_LOSS
                        To print losses after each step
--log-every-step LOG_EVERY_STEP
                        Tensorboard logging every X steps
--max-checkpoints MAX_CHECKPOINTS
                        The maximum number of checkpoints.
--save-model-frequency SAVE_MODEL_FREQUENCY
                        Save model every X epochs.
--checkpoint-frequency CHECKPOINT_FREQUENCY
                        Save checkpoints every X epochs.
```

### Training Optimizations

[Mixed precision](https://www.tensorflow.org/guide/mixed_precision) training is supported. However, your GPU must have compute capability of at least 7.0.

To check your GPU's compute capability, run the following command:

```cmd
D:\>nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 511.65       Driver Version: 511.65       CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   49C    P0    20W /  N/A |    409MiB /  6144MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

Finetuning can be accomplished by setting the `--from-checkpoint` flag to True.

### Tensorboard

To view Tensorboard, run the following command:

```cmd
tensorboard --logdir=<path_to_training_dir/tensorboard>
```

## Testing

To test the trained model run the following command.

```cmd
python inference.py --testing-image-dir <path_to_testing_images_dir> \
                    --save-image-dir <path_to_save_dir> \
                    --model <path_to_model> \
                    --labels-file <path_to_labels_file> \
                    --score-threshold 0.35 \
                    --iou-threshold 0.5
```

### All commands for inference

```cmd
(venv) D:\EfficientDet>python inference.py -h
usage: i-Sight [-h] [--testing-image-dir TESTING_IMAGE_DIR]
            [--save-image-dir SAVE_IMAGE_DIR] [--model-dir MODEL_DIR]
            [--image-dims IMAGE_DIMS] [--labels-file LABELS_FILE]
            [--score-threshold SCORE_THRESHOLD] [--iou-threshold IOU_THRESHOLD]

Run i-Sight Tests

optional arguments:
-h, --help            show this help message and exit
--testing-image-dir TESTING_IMAGE_DIR
                        Path to testing images directory.
--save-image-dir SAVE_IMAGE_DIR
                        Path to testing images directory.
--model-dir MODEL_DIR
                        Path to testing model directory.
--image-dims IMAGE_DIMS
                        Size of the input image.
--labels-file LABELS_FILE
                        Path to labels file.
--score-threshold SCORE_THRESHOLD
                        Score threshold for NMS.
--iou-threshold IOU_THRESHOLD
                        IOU threshold for NMS.
```

## Pretrained Weights

Models were trained on PascalVOC2012, tf saved models they can only be used for inference. Saved weights can be used for transfer learning. More models and weights to come soon!

| Model | Model | Weights |
| --- | --- | --- |
| EfficientDet-D0 | [Link](https://drive.google.com/file/d/14TJdevtt8sG9gCJJ0aovO04hDvsZqOrr/view?usp=sharing) | [Link](https://drive.google.com/file/d/1X4rmnL7QKlM6f_xuqUaDPPEAXDI8JbOE/view?usp=sharing) |

## References

[1] [Official Repo](https://github.com/google/automl/tree/master/efficientdet)
[2] [Paper](https://arxiv.org/abs/1906.02768)
[3] [https://github.com/joydeepmedhi/Anchor-Boxes-with-KMeans]
[4] [https://github.com/fizyr/keras-retinanet]

## License

```License
Copyright 2020 Thomas Chia (IdeaKing)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
