import os
import shutil
import tensorflow as tf

from typing import List, Tuple


class Train:
    def __init__(self,
                 training_dir: str,
                 epochs: int,
                 total_steps: int,
                 input_shape: Tuple[int, int] = (512, 512),
                 precision: str = "float32",
                 max_checkpoints: int = 10,
                 checkpoint_frequency: int = 10,
                 save_model_frequency: int = 10,
                 print_loss: bool = True,
                 log_every_step: int = 100,
                 from_checkpoint: bool = False):
        """ Trains the model.

        Params:
            training_dir (str): The filepath to the training directory
            epochs (int): The number of epochs to train the model
            total_steps (int): The total number of steps
            precision (str): Can either be "float32" or "mixed_float16"
            max_checkpoints (int): The total number of checkpoints to save
        """
        # Initialize the directories
        if os.path.exists(training_dir) and from_checkpoint == False:
            # Prevents accidental deletions
            input("Press Enter to delete the current directory and continue.")
            shutil.rmtree(training_dir)
        else:
            os.makedirs(training_dir)

        # Tensorboard Logging
        tensorboard_dir = os.path.join(
            training_dir, "tensorboard")
        if os.path.exists(tensorboard_dir) is False:
            os.makedirs(tensorboard_dir)
        tensorboard_file_writer = tf.summary.create_file_writer(
            tensorboard_dir)
        tensorboard_file_writer.set_as_default()

        # Define the checkpoint directories
        self.checkpoint_dir = os.path.join(
            training_dir, "model")
        # Define the full model directories
        self.exported_dir = os.path.join(
            training_dir, "model-exported")

        self.epochs = epochs
        self.total_steps = total_steps
        self.steps_per_epoch = int(self.total_steps/self.epochs)
        self.input_shape = input_shape
        self.precision = precision
        self.max_checkpoints = max_checkpoints
        self.checkpoint_frequency = checkpoint_frequency
        self.save_model_frequency = save_model_frequency
        self.print_loss = print_loss
        self.log_every_step = log_every_step
        self.from_checkpoint = from_checkpoint

    def supervised(self,
                   dataset: tf.data.Dataset,
                   model: tf.keras.models.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   losses: List or tf.keras.losses.Loss):
        """Supervised training on the model."""
        # Checkpointing Functions
        checkpoint = tf.train.Checkpoint(
            model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            self.checkpoint_dir,
            self.max_checkpoints)
        
        if self.from_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print("Restored from checkpoint: {}".format(
                checkpoint_manager.latest_checkpoint))

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                # Run losses
                if isinstance(losses, list):
                    loss = 0
                    for loss_func in losses:
                        loss += loss_func(y_true=y, y_pred=preds)
                else:
                    loss = losses(y_true=y, y_pred=preds)
                if self.precision == "mixed_float16":
                    loss = optimizer.get_scaled_loss(loss)
            gradients = tape.gradient(
                target=loss,
                sources=model.trainable_variables)
            if self.precision == "mixed_float16":
                gradients = optimizer.get_unscaled_gradients(gradients)
            optimizer.apply_gradients(
                grads_and_vars=zip(gradients, model.trainable_variables))
            return loss

        global_step = 0
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            for step, (images, label_cls, label_bbx) in enumerate(dataset):
                labels = (label_cls, label_bbx)
                loss = train_step(images, labels)
                if self.print_loss:
                    print(f"Epoch {epoch} Step {step}/{self.steps_per_epoch}", \
                            f"loss {loss}")
                if global_step % self.checkpoint_frequency == 0:
                    checkpoint_manager.save()
                global_step = global_step + 1

            if epoch % self.save_model_frequency == 0:
                tf.keras.models.save_model(
                    model, self.exported_dir)

        print("Finished training.")
        return model
