# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz

import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.dsl import PipelineVolume


def volume_op():
    return dsl.VolumeOp(
        name="create pipeline volume",
        resource_name="pipeline-pvc",
        modes=["ReadWriteOnce"],
        size="3Gi",
    )


def preprocess_op(pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='preprocessing',
        image='yonsweng/kubeflow-mnist-preprocessing',
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

    preprocess_result = preprocess_op(pvolume=volumn_result.volume, data_dir=data_dir)

    train_result = train_op(pvolume=preprocess_result.pvolume, data_dir=data_dir, model_dir=model_dir)


if __name__ == '__main__':
    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
