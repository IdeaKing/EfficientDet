import argparse

parser = argparse.ArgumentParser(
    description="Run EfficientDet Training or Tests",
    prog="EfficientDet")

# Directory Configurations
parser.add_argument("--dataset-path",
                    type=str,
                    default="data/datasets/VOC2012",
                    help="Path to dataset")
parser.add_argument("--training-dir",
                    type=str,
                    default="training_dir/VOC-2012",
                    help="Path to the training directory")

# Model Configurations
parser.add_argument("--model",
                    type=str,
                    default="efficientdet_d0",
                    help="Model name, can be \
                          efficientdet_d[0 - 7] (object detection)")

# Training Configurations
parser.add_argument("--debug",
                    type=bool,
                    default=False,
                    help="Use debugging mode or not, \
                          NOTE: SEVERELY REDUCES PPERFORMANCE")
parser.add_argument("--precision",
                    type=str,
                    default="float32",
                    help="The precision type, can either be \
                          mixed_float16 or float32")
parser.add_argument("--batch-size",
                    type=int,
                    default=8,
                    help="Batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=300,
                    help="The number of epochs to train the model(s)")
parser.add_argument("--optimizer",
                    type=str,
                    default="SGD",
                    help="The optimizer to train the model on, can be: \
                          SGD or ADAM")
parser.add_argument("--learning-rate",
                    type=float,
                    default=1e-4,
                    help="The initial learning rate for the optimizer")
parser.add_argument("--optimizer-momentum",
                    type=float,
                    default=0.9,
                    help="The momentum for the optimizer")
parser.add_argument("--from-checkpoint",
                    type=bool,
                    default=False,
                    help="Continue training from checkpoint.")

# Dataset Configurations
parser.add_argument("--dataset-files",
                    default="labeled_train.txt",
                    help="Either filename of labeled_train.txt")
parser.add_argument("--labels-file",
                    default="labels.txt",
                    help="Filename of the labels file.")
parser.add_argument("--images-dir",
                    default="images",
                    help="Folder name that holds images.")
parser.add_argument("--labels-dir",
                    default="labels",
                    help="Folder name that holds labels.")
parser.add_argument("--shuffle-size",
                    type=int,
                    default=16,
                    help="Shuffle the dataset steps, keep in powers of 2")
parser.add_argument("--image-dims",
                    type=tuple,
                    default=(512, 512),
                    help="Images dims to fit in the model")
parser.add_argument("--augment-ds",
                    type=bool,
                    default=True,
                    help="If augmentation is needed")

# Logging
parser.add_argument("--print-loss",
                    type=bool,
                    default=True,
                    help="To print losses after each step")
parser.add_argument("--log-every-step",
                    type=int,
                    default=100,
                    help="Tensorboard logging every X steps")

# Checkpointing and Saving
parser.add_argument("--max-checkpoints",
                    type=int,
                    default=10,
                    help="The maximum number of checkpoints.")
parser.add_argument("--save-model-frequency",
                    type=int,
                    default=10,
                    help="Save model every X epochs.")
parser.add_argument("--checkpoint-frequency",
                    type=int,
                    default=10,
                    help="Save checkpoints every X epochs.")

args = parser.parse_args()
