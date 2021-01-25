# Running Locally
```
% python preprocessing.py --data_dir data
% python train.py --data_dir data --model_path export
```

# Building the image

DOCKER_BUILDKIT=1 docker build -t benjamintanweihao/kubeflow-mnist env -f Dockerfile

# Tensorflow Serving

docker run -t --rm -p 8501:8501 \
    -v "$PWD/export:/models/mnist" \
    -e MODEL_NAME=mnist \
    tensorflow/serving:1.14.0