import kfp.dsl as dsl
from kfp.dsl import PipelineVolume

# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz
from constants import PROJECT_ROOT, CONDA_PYTHON_CMD


def git_clone_darkrai_op(repo_url: str):
    image = 'alpine/git:latest'

    commands = [
        f"git clone {repo_url} {PROJECT_ROOT}",
        f"cd {PROJECT_ROOT}"]

    volume_op = dsl.VolumeOp(
        name="create pipeline volume",
        resource_name="pipeline-pvc",
        modes=["ReadWriteOnce"],
        size="3Gi"
    )

    op = dsl.ContainerOp(
        name='git clone',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": volume_op.volume}
    )

    return op


def preprocess_op(image: str, pvolume: PipelineVolume, data_dir: str):
    arguments = f"{CONDA_PYTHON_CMD} {PROJECT_ROOT}/preprocessing.py --data_dir={data_dir}"

    op = dsl.ContainerOp(
        name='preprocessing',
        image=image,
        command=['bash'],
        arguments=['-c', arguments],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


def train_and_eval_op(image: str, pvolume: PipelineVolume, data_dir: str, ):
    return dsl.ContainerOp(
        name='training and evaluation',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/train.py"],
        arguments=["--data_dir", data_dir],
        file_outputs={'output': f'{PROJECT_ROOT}/output.txt'},
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


def export_saved_model_op(image: str, pvolume: PipelineVolume, model_dir: str):
    return dsl.ContainerOp(
        name='export saved model',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/export.py"],
        arguments=["--model_dir", model_dir],
        file_outputs={'output': '/workspace/output.txt'},
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


def optimizer_op(image: str, pvolume: PipelineVolume, model_dir: str):
    return dsl.ContainerOp(
        name='optimizer',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/optimise.py"],
        arguments=[
            '--model_dir', model_dir
        ],
        file_outputs={'output': '/workspace/output.txt'},
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


@dsl.pipeline(
    name='Fashion MNIST Training Pipeline',
    description='Fashion MNIST Training Pipeline to be executed on KubeFlow.'
)
def training_pipeline(image: str, repo_url: str, data_dir: str):
    git_clone = git_clone_darkrai_op(repo_url=repo_url)

    preprocess_data = preprocess_op(image=image,
                                    pvolume=git_clone.pvolume,
                                    data_dir=data_dir)

    _training_and_eval = train_and_eval_op(image=image,
                                           pvolume=preprocess_data.pvolume,
                                           data_dir=data_dir)


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
