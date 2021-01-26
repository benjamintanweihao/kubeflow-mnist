# First launch Docker
# docker run -t --rm -p 8501:8501 \
#                            -v "$PWD/export:/models/mnist" \
#                               -e MODEL_NAME=mnist \
#     tensorflow/serving:1.14.0

# Grab a sample image.
import os
import pickle
import random
import matplotlib.pyplot as plt
import requests
import json
import numpy as np


def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28, 28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()


with open(os.path.join('data', 'test_images.pickle'), 'rb') as f:
    test_images = pickle.load(f)

with open(os.path.join('data', 'test_labels.pickle'), 'rb') as f:
    test_labels = pickle.load(f)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# rando = random.randint(0, len(test_images) - 1)
# show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))

# https://www.tensorflow.org/tfx/serving/api_rest

headers = {"content-type": "application/json"}
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
json_response = requests.post('http://localhost:8503/v1/models/mnist:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))
