import os
import sys
from os.path import join, exists

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import Adam
from utils.data_loader import generator_fn
from utils.util import AnchorParameters
from nets.nn import build_model, regression_loss, classification_loss
from nets.helper import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from utils import config

strategy = tf.distribute.MirroredStrategy()

f_names = []
with open(join(config.data_dir, 'labeled_train.txt')) as reader:
    for line in reader.readlines():
        f_names.append(line.rstrip().split(' ')[0])
np.random.shuffle(f_names)
dataset = tf.data.Dataset.from_generator(generator_fn(f_names), (tf.float32, tf.float32, tf.float32))
dataset = dataset.batch(config.batch_size * strategy.num_replicas_in_sync)
dataset = strategy.experimental_distribute_dataset(dataset)

num_replicas = strategy.num_replicas_in_sync

model, prediction_model = build_model(config.phi,
                                        num_classes=len(config.classes),
                                        num_anchors=AnchorParameters().num_anchors())
if config.weight_path == 'imagenet':
    model_name = 'efficientnet-b{}'.format(config.phi)
    file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
    file_hash = WEIGHTS_HASHES[model_name][1]
    weights_path = get_file(file_name,
                            BASE_WEIGHTS_PATH + file_name,
                            cache_subdir='models',
                            file_hash=file_hash)
    model.load_weights(weights_path, by_name=True)
    for i in range(1, [227, 329, 329, 374, 464, 566, 656][config.phi]):
        model.layers[i].trainable = False
    optimizer = Adam(1e-3)
else:
    print('Loading model, this may take a second...')
    model.load_weights(config.weight_path)
    optimizer = Adam(5e-5)

c_loss_object = classification_loss()
r_loss_object = regression_loss()


def compute_loss(label, box, y_pred):
    c_loss = c_loss_object(label, y_pred[0])
    r_loss = r_loss_object(box, y_pred[1])
    total_loss = c_loss + r_loss
    return tf.divide(tf.reduce_sum(total_loss), tf.constant(num_replicas, dtype=tf.float32))

def train_step(image, label, box):
    with tf.GradientTape() as tape:
        y_pred = model(image)

        loss = compute_loss(label, box, y_pred)
    train_variable = model.trainable_variables
    gradient = tape.gradient(loss, train_variable)
    optimizer.apply_gradients(zip(gradient, train_variable))
    return loss

@tf.function
def distribute_train_step(image, label, box):
    loss = train_step(image, label, box)
    return loss


def main():
    if not exists(join('weights', f'D{config.phi}')):
        os.makedirs(join('weights', f'D{config.phi}'))
    with open(join('weights', 'history.txt'), 'w') as writer:
        print(f"--- Training with {config.steps} Steps ---")
        for step, inputs in enumerate(dataset):
            step += 1
            image, label, box = inputs
            loss = distribute_train_step(image, label, box)
            loss = f'{loss.numpy():.8f}'
            print(f'[{step}] - {loss}')
            writer.write(f'{step}\t{loss}\n')
            if step % config.steps == 0:
                model.save_weights(join("weights", f'D{config.phi}', f'model{step // config.steps}.h5'))
            if step // config.steps == config.epochs:
                sys.exit("--- Stop Training ---")


if __name__ == '__main__':
    main()
