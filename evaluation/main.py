import calendar
import os
import time

from tensorflow import keras
import tensorflow as tf
import pickle
import argparse


def evaluate(data_dir, model_path):
    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'rb') as f:
        test_images = pickle.load(f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'rb') as f:
        test_labels = pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow FMNIST evaluation script')
    parser.add_argument('--data-dir', help='path to images and labels.')
    parser.add_argument('--model-path', help='folder to export model')
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_path)
