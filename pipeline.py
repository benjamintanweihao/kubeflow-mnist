# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz

import kfp
import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.dsl import PipelineVolume
from kfp.components import func_to_container_op


def volume_op():
    return dsl.VolumeOp(
        name="create pipeline volume",
        resource_name="pipeline-pvc",
        modes=["ReadWriteOnce"],
        size="3Gi",
    )


@func_to_container_op
def flip_coin_op() -> str:
    """Flip a coin and output heads or tails randomly."""
    import random
    result = random.choice(['heads', 'tails'])
    print(result)
    return result


def preprocess_op(pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='preprocessing norm',
        image='yonsweng/kubeflow-mnist-preprocessing',
        command=["python", "main.py"],
        arguments=["--data-dir", data_dir],
        pvolumes={data_dir: pvolume},
    )


def preprocess_op_std(pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='preprocessing std',
        image='yonsweng/kubeflow-mnist-preprocessing-std',
        command=["python", "main.py"],
        arguments=["--data-dir", data_dir],
        pvolumes={data_dir: pvolume},
    )


def train_op(pvolume: PipelineVolume, data_dir: str, model_dir: str):
    return dsl.ContainerOp(
        name='training',
        image='yonsweng/kubeflow-mnist-training',
        command=["python", "main.py"],
        arguments=["--data-dir", data_dir, "--model-dir", model_dir],
        file_outputs={'output': 'output.txt'},
        pvolumes={data_dir: pvolume}
    )


@dsl.pipeline(
    name='Fashion MNIST Training Pipeline',
    description='Fashion MNIST Training Pipeline to be executed on KubeFlow.'
)
def training_pipeline(data_dir: str = '/data/', model_dir: str = '/data/models/'):
    volumn_result = volume_op()

    flip = flip_coin_op()
    with dsl.Condition(flip.output == 'heads'):
        preprocess_result = preprocess_op_std(pvolume=volumn_result.volume, data_dir=data_dir)
    with dsl.Condition(flip.output == 'tails'):
        preprocess_result = preprocess_op(pvolume=volumn_result.volume, data_dir=data_dir)

    train_result = train_op(pvolume=preprocess_result.pvolume, data_dir=data_dir, model_dir=model_dir)


if __name__ == '__main__':
    # kfp.Client(host='http://localhost:8080').create_run_from_pipeline_func(
    #     training_pipeline, arguments={})
    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
