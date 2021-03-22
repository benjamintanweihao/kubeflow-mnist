import argparse
from kubernetes import client

from kfserving import KFServingClient
from kfserving import constants
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2TensorflowSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kubernetes.client import V1ResourceRequirements


def create_inference_service(namespace: str,
                             name: str,
                             storage_uri: str,
                             runtime_version: str,
                             service_account_name: str):
    api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION
    default_endpoint_spec = V1alpha2EndpointSpec(
        predictor=V1alpha2PredictorSpec(
            min_replicas=1,
            service_account_name=service_account_name,
            tensorflow=V1alpha2TensorflowSpec(
                runtime_version=runtime_version,
                storage_uri=storage_uri,
                resources=V1ResourceRequirements(
                    requests={'cpu': '100m', 'memory': '1Gi'},
                    limits={'cpu': '100m', 'memory': '1Gi'}))))

    isvc = V1alpha2InferenceService(api_version=api_version,
                                    kind=constants.KFSERVING_KIND,
                                    metadata=client.V1ObjectMeta(
                                        name=name, namespace=namespace),
                                    spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))
    KFServing = KFServingClient()
    KFServing.create(isvc)
    KFServing.get(name, namespace=namespace, watch=True, timeout_seconds=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--namespace",
                        help="namespace to deploy the inference service to",
                        type=str,
                        default='kfserving-inference-service')

    parser.add_argument("--name",
                        help="name of inference service",
                        default='fashion-mnist',
                        type=str)

    parser.add_argument("--storage_uri",
                        help="storage_uri of model. e.g. s3://servedmodels/ssd_inception_v2_coco/1",
                        default="s3://servedmodels/fmnist/saved_models/1611590079",
                        type=str)

    parser.add_argument("--runtime_version",
                        help="version of Tensorflow. e.g. 1.14.0 or 1.14.0-gpu",
                        type=str,
                        default='1.14.0')

    parser.add_argument("--service_account_name",
                        help="Service Account name that stores the credentials",
                        type=str,
                        default='sa')

    args = parser.parse_args()

    create_inference_service(namespace=args.namespace,
                             name=args.name,
                             storage_uri=args.storage_uri,
                             runtime_version=args.runtime_version,
                             service_account_name=args.service_account_name)
