from tensorflow import keras
import os
import tensorflow as tf
import pickle

# Download

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

with open('/tmp/train_images.pickle', 'wb') as f:
    pickle.dump(train_images, f)

with open('/tmp/train_labels.pickle', 'wb') as f:
    pickle.dump(train_labels, f)

with open('/tmp/test_images.pickle', 'wb') as f:
    pickle.dump(test_images, f)

with open('/tmp/test_labels.pickle', 'wb') as f:
    pickle.dump(test_labels, f)

# Training
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

with open('/tmp/train_images.pickle', 'rb') as f:
    train_images = pickle.load(f)

with open('/tmp/train_labels.pickle', 'rb') as f:
    train_labels = pickle.load(f)

model.fit(train_images, train_labels, epochs=10)

with open('/tmp/test_images.pickle', 'rb') as f:
    test_images = pickle.load(f)

with open('/tmp/test_labels.pickle', 'rb') as f:
    test_labels = pickle.load(f)

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

model_dir = '/home/benjamintan/dev/kubeflow-mnist/'
# Save model
tf.saved_model.save(model, model_dir)

