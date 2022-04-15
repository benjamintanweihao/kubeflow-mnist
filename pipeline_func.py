import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp.dsl import InputPath, OutputPath
from kfp.components import create_component_from_func


# @component(
#     base_image='tensorflow/tensorflow:1.14.0-gpu-py3',
#     packages_to_install=[
#         'kfp-pipeline-spec==0.1.13',
#         'six==1.13.0',
#     ],
# )
def preprocess_op(data_dir: OutputPath()):
    import os
    import pickle
    import tensorflow as tf

    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'train_images.pickle'), 'wb') as f:
        pickle.dump(train_images, f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'wb') as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'wb') as f:
        pickle.dump(test_images, f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'wb') as f:
        pickle.dump(test_labels, f)


# @component(
#     base_image='tensorflow/tensorflow:1.14.0-gpu-py3',
#     packages_to_install=[
#         'kfp-pipeline-spec==0.1.13',    
#         'six==1.13.0',
#     ],
# )
def train_op(data_dir: InputPath()):
    import os
    import pickle
    import tensorflow as tf

    # Training
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    with open(os.path.join(data_dir, 'train_images.pickle'), 'rb') as f:
        train_images = pickle.load(f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'rb') as f:
        train_labels = pickle.load(f)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'rb') as f:
        test_images = pickle.load(f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'rb') as f:
        test_labels = pickle.load(f)

    model.fit(train_images, train_labels, epochs=10)

    # Evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f'Test Loss: {test_loss}')
    print(f'Test Acc: {test_acc}')


@dsl.pipeline(
    name='Fashion MNIST Training Pipeline',
    description='Fashion MNIST Training Pipeline to be executed on KubeFlow.',
)
def training_pipeline():
    preprocess = create_component_from_func(
        preprocess_op,
        base_image='tensorflow/tensorflow:1.14.0-gpu-py3',
    )
    train = create_component_from_func(
        train_op,
        base_image='tensorflow/tensorflow:1.14.0-gpu-py3',
    )

    preprocess_result = preprocess()

    train_result = train_op(preprocess_result.output)


if __name__ == '__main__':
    kfp.Client(host='http://localhost:8080').create_run_from_pipeline_func(
        training_pipeline, arguments={})
    # compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
