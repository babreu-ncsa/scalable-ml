# external imports
import tensorflow as tf
from tensorflow import keras
# built-in imports
import time
import os
import argparse
# local imports
import mpi_workers as mpiw

def get_compiled_model(n_inter_layers, n_units):
    """
    Compiles a keras model with n_inter_layers Dense layers with n_units each and ReLu activation.
    """
    inputs = keras.Input(shape=(784,))
    layer_size=n_units

    x = keras.layers.Dense(layer_size, activation="relu")(inputs)
    for _ in range(n_inter_layers - 1):
        x = keras.layers.Dense(layer_size, activation="relu")(x)

    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def get_dataset(batch_size):
    """
    Gets MNIST dataset from Keras. Returns the batched dataset.

    Arguments:
        - batch_size: the batch size

    Returns:
        - 3-entry tuple with train, validation, and test datasets in Tensorflow.data.Dataset format.
    """
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


def train_model_on_gpus(epochs, batch_size, n_layers, n_units):
    """
    Trains a model with n_layers Dense layers with n_units each and ReLu activation, using batch_size as the batch size.
    Arguments:
        - epochs: number of epochs to train the model
        - batch_size: the batch size
        - n_layers: number of hidden layers in the neural network
        - n_units: number of units in each hidden layer
    """
    # set TF_CONFIG across all nodes
    mpiw.set_tf_config()

    # find devices on each node
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(
          device_type)

    # define the training strategy for multi node training
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # compile model with that strategy
    with strategy.scope():
        model = get_compiled_model(n_layers, n_units)

    # get the dataset
    train_dataset, val_dataset, test_dataset = get_dataset(batch_size)

    # train the model
    start = time.perf_counter()
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=0)
    end = time.perf_counter()

    # print training stats
    task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id
    )
    if mpiw._is_chief(task_type, task_id):
        print(model.summary())
        print('training time (s): %.3f\n' % (end-start))

    return

def parse_input():
    """
    Parses the input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Tensorflow Multi-node MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Input batch size to be distributed across all nodes',
    )

    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help='Number of hidden layers in the neural network',
    )

    parser.add_argument(
        '--n_units',
        type=int,
        default=128,
        help='Number of units in each hidden layer',
    )

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=2,
        help='Number of epochs to train the model',
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_input()
    train_model_on_gpus(
        args.n_epochs, 
        args.batch_size, 
        args.n_layers, 
        args.n_units
    )


