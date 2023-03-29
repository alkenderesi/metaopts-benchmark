"""Metaopts evaluation."""

import numpy as np
import tensorflow as tf
import metaopts
import sys
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from benchmark import mp_model, dnn_model, cnn_model


def main():

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    result_path = os.path.normpath(sys.argv[1])

    for model_type in os.listdir(result_path):

        if model_type == 'mp': model = mp_model()
        elif model_type == 'dnn': model = dnn_model()
        elif model_type == 'cnn': model = cnn_model()
        else: print('No model named: {0}'.format(model_type), file=sys.stderr); continue

        model_result_path = os.path.join(result_path, model_type)
        model_files = os.listdir(model_result_path)
        pickle_files = [file for file in model_files if file.endswith('.pickle')]

        with open('{0}/{1}_evaluation.csv'.format(model_result_path, model_type), 'w') as evaluation_file:

            evaluation_file.write('algorithm,accuracy,precision,recall,f1score\n')

            for pickle_file in pickle_files:

                pickled_weights = metaopts.utilities.load_individual(os.path.join(model_result_path, pickle_file))

                for mtw, pw in zip(model.trainable_weights, pickled_weights):
                    mtw.assign(pw)

                y_pred = np.argmax((model(x_test)).numpy(), axis=1)

                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1score, _ = precision_recall_fscore_support(
                    y_test, 
                    y_pred, 
                    average='macro', 
                    zero_division=0
                )

                evaluation_file.write('{0},{1},{2},{3},{4}\n'.format(
                    pickle_file[:-15], 
                    accuracy, 
                    precision, 
                    recall, 
                    f1score
                ))


if __name__ == '__main__':
    main()
