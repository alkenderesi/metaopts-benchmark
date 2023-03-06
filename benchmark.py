"""Metaopts benchmarks."""

import tensorflow as tf
import metaopts
import datetime
import pickle
import yaml
import sys


def mp_model():
    """
    Returns a Multiclass Perceptron model.
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def dnn_model():
    """
    Returns a Deep Neural Network model.
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def cnn_model():
    """
    Returns a Convolutional Neural Network model.
    """
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


def gradient_training(
        model,
        loss_fn,
        x_train,
        y_train,
        optimizer,
        config
    ):
    """
    Starts a gradient based training loop.

    Args:
        model: `tf.keras.Model` - Neural network model.
        loss_fn: `tf.keras.losses.Loss` - Loss function.
        x_train: `list` - Input data.
        y_train: `list` - Target data.
        optimizer: `str` - Short name of the optimizer algorithm.
        config: `dict` - Benchmark configuration.
    """

    loss_cache = []

    def record_loss(epoch, logs):
        now = datetime.datetime.now()
        date = now.strftime('%Y,%m,%d')
        time = now.strftime('%H,%M,%S,%f')
        loss_cache.append('{0},{1},{2}'.format(date, time, logs['loss']))

    callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_end=record_loss)]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )

    model.fit(
        x_train,
        y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=callbacks
    )

    with open('{0} fitness.csv'.format(optimizer), 'w') as log_file:
        for loss in loss_cache:
            log_file.write('{0}\n'.format(loss))
    
    with open('{0} weights.pickle'.format(optimizer), 'wb') as save_file:
        pickle.dump(model.trainable_weights, save_file)


def metaheuristic_training(
        model,
        loss_fn,
        x_train,
        y_train,
        optimizer,
        config
    ):
    """
    Starts a metaheuristic based training loop.

    Args:
        model: `tf.keras.Model` - Neural network model.
        loss_fn: `tf.keras.losses.Loss` - Loss function.
        x_train: `list` - Input data.
        y_train: `list` - Target data.
        optimizer: `str` - Short name of the optimizer algorithm.
        config: `dict` - Benchmark configuration.
    """

    fitness_fn = metaopts.create_fitness_function(
        model,
        loss_fn,
        tf.constant(x_train),
        tf.constant(y_train),
        config['batch_size']
    )

    if optimizer == 'ga':
        metaopts.ga(
            model.trainable_weights,
            fitness_fn,
            config['generation_limit'],
            config['fitness_limit'],
            config['population_size'],
            config['ga']['elite_size'],
            config['transfer_learning'],
            config['fitness_log_frequency'],
            config['best_individual_save_frequency'],
            config['ga']['learning_rate'],
            config['ga']['crossover_rate'],
            config['ga']['mutation_rate']
        )
    
    if optimizer == 'avoa':
        metaopts.avoa(
            model.trainable_weights,
            fitness_fn,
            config['generation_limit'],
            config['fitness_limit'],
            config['population_size'],
            config['transfer_learning'],
            config['fitness_log_frequency'],
            config['best_individual_save_frequency'],
            config['avoa']['L1'],
            config['avoa']['L2'],
            config['avoa']['w'],
            config['avoa']['P1'],
            config['avoa']['P2'],
            config['avoa']['P3'],
            config['avoa']['lb'],
            config['avoa']['ub'],
            config['avoa']['beta']
        )

    if optimizer == 'mvo':
        metaopts.mvo(
            model.trainable_weights,
            fitness_fn,
            config['generation_limit'],
            config['fitness_limit'],
            config['population_size'],
            config['transfer_learning'],
            config['fitness_log_frequency'],
            config['best_individual_save_frequency'],
            config['mvo']['min'],
            config['mvo']['max'],
            config['mvo']['p'],
            config['mvo']['lower_bound'],
            config['mvo']['upper_bound']
        )
    
    if optimizer == 'dgo':
        metaopts.dgo(
            model.trainable_weights,
            fitness_fn,
            config['generation_limit'],
            config['fitness_limit'],
            config['population_size'],
            config['transfer_learning'],
            config['fitness_log_frequency'],
            config['best_individual_save_frequency'],
            config['dgo']['throw_count']
        )
    
    if optimizer == 'stbo':
        metaopts.stbo(
            model.trainable_weights,
            fitness_fn,
            config['generation_limit'],
            config['fitness_limit'],
            config['population_size'],
            config['transfer_learning'],
            config['fitness_log_frequency'],
            config['best_individual_save_frequency'],
            config['stbo']['lb'],
            config['stbo']['ub']
        )


def main():
    
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if sys.argv[1] == 'mp': model = mp_model()
    elif sys.argv[1] == 'dnn': model = dnn_model()
    elif sys.argv[1] == 'cnn': model = cnn_model()
    else: print('No model named: {0}'.format(sys.argv[1]), file=sys.stderr); return

    if sys.argv[2] in ('sgd', 'adam'):
        gradient_training(
            model,
            loss_fn,
            x_train,
            y_train,
            sys.argv[2],
            config
        )
    elif sys.argv[2] in ('ga', 'avoa', 'mvo', 'dgo', 'stbo'):
        metaheuristic_training(
            model,
            loss_fn,
            x_train,
            y_train,
            sys.argv[2],
            config
        )
    else: print('No algorithm named: {0}'.format(sys.argv[2]), file=sys.stderr); return


if __name__ == '__main__':
    main()
