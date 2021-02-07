import kfp
from kfp import dsl
from kfp.dsl import PipelineVolume

from constants import CONDA_PYTHON_CMD, PROJECT_ROOT


def git_clone_op(repo_url: str):
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


def serving_op(image: str,
               pvolume: PipelineVolume,
               bucket_name: str,
               model_name: str,
               model_version: str):
    namespace = 'kfserving-inference-service'
    runtime_version = '1.14.0'
    service_account_name = 'sa'

    storage_uri = f"s3://{bucket_name}/{model_name}/saved_models"

    op = dsl.ContainerOp(
        name='serve model',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/serving/kfs_deployer.py"],
        arguments=[
            '--namespace', namespace,
            '--name', f'{model_name}-{model_version}-1',
            '--storage_uri', storage_uri,
            '--runtime_version', runtime_version,
            '--service_account_name', service_account_name
        ],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


@dsl.pipeline(
    name='Serving Pipeline',
    description='This is a single component Pipeline for Serving'
)
def serving_pipeline(
        image: str = 'benjamintanweihao/kubeflow-mnist',
        repo_url: str = 'https://github.com/benjamintanweihao/kubeflow-mnist.git',
):
    model_name = 'fmnist'
    export_bucket = 'servedmodels'
    model_version = '1611590079'

    git_clone = git_clone_op(repo_url=repo_url)

    serving_op(image=image,
               pvolume=git_clone.pvolume,
               bucket_name=export_bucket,
               model_name=model_name,
               model_version=model_version)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(serving_pipeline, 'serving-pipeline.zip')
