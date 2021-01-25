import calendar
import os
import time

from tensorflow import keras
import tensorflow as tf
import pickle
import argparse

from constants import PROJECT_ROOT


def train(data_dir: str):
    # Training
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    with open(os.path.join(data_dir, 'train_images.pickle'), 'rb') as f:
        train_images = pickle.load(f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'rb') as f:
        train_labels = pickle.load(f)

    model.fit(train_images, train_labels, epochs=10)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'rb') as f:
        test_images = pickle.load(f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'rb') as f:
        test_labels = pickle.load(f)

    # Evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f'Test Loss: {test_loss}')
    print(f'Test Acc: {test_acc}')

    # Save model
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = PROJECT_ROOT

    ts = calendar.timegm(time.gmtime())
    model_path = os.path.join(model_path, str(ts))
    tf.saved_model.save(model, model_path)

    with open(os.path.join(PROJECT_ROOT, 'output.txt'), 'w') as f:
        f.write(model_path)
        print(f'Model written to: {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow FMNIST training script')
    parser.add_argument('--data_dir', help='path to images and labels.')
    parser.add_argument('--model_path', help='folder to export model')
    args = parser.parse_args()

    train(data_dir=args.data_dir)
