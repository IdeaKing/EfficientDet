"""
Main training function for EfficientDet.

Thomas Chia 2022
"""

import os
import tensorflow as tf

from utils import arg_parser, file_reader
from models import efficientdet
from losses import loss
import dataset, train


def main(args):
    if args.debug:
        tf.config.run_functions_eagerly(True)
    # Set the precision
    if args.precision != ("mixed_float16" or "float32"):
        ValueError(f"{args.precision} is not a precision type.")
    tf.keras.mixed_precision.set_global_policy(args.precision)

    file_names = dataset.load_data(dataset_path=args.dataset_path,
                                    file_name=args.dataset_files)
    total_steps = int((len(file_names) / args.batch_size) * args.epochs)
    labels_dict = file_reader.parse_label_file(
        path_to_label_file=os.path.join(args.dataset_path, args.labels_file))
    num_classes = len(labels_dict)
    train_func = train.Train(training_dir=args.training_dir,
                             epochs=args.epochs,
                             total_steps=total_steps,
                             input_shape=args.image_dims,
                             precision=args.precision,
                             max_checkpoints=args.max_checkpoints,
                             checkpoint_frequency=args.checkpoint_frequency,
                             save_model_frequency=args.save_model_frequency,
                             print_loss=args.print_loss,
                             log_every_step=args.log_every_step,
                             from_checkpoint=args.from_checkpoint)
    if args.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                            momentum=args.optimizer_momentum)
    elif args.optimizer == "ADAM":
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_steps=int(0.01 * total_steps),
            decay_rate=0.96,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)    
    else:
        raise ValueError(f"{args.optimizer} is not an available optimizer")

    dataset_creater = dataset.Dataset(file_names=file_names,
                                        dataset_path=args.dataset_path,
                                        labels_dict=labels_dict,
                                        batch_size=args.batch_size,
                                        shuffle_size=args.shuffle_size,
                                        images_dir=args.images_dir,
                                        labels_dir=args.labels_dir,
                                        image_dims=args.image_dims,
                                        augment_ds=args.augment_ds)
    labeled_ds = dataset_creater()
    model = efficientdet.get_efficientdet(name=args.model,
                                            input_shape=args.image_dims,
                                            num_classes=num_classes)
    loss_func = loss.EffDetLoss(num_classes=num_classes)
    trained_model = train_func.supervised(dataset=labeled_ds,
                                            model=model,
                                            optimizer=optimizer,
                                            losses=loss_func)
    
    tf.keras.models.save_model(
        trained_model, 
        os.path.join(args.training_dir, "final-model"))

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    args = arg_parser.args
    main(args)
