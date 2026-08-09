docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 . -t mat

docker run --gpus all --network host -it   -e DISPLAY=$DISPLAY   -e XAUTHORITY=/tmp/.docker.xauth   -v $HOME/.Xauthority:/tmp/.docker.xauth:ro   mat bash

docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:2.13.0-gpu . -t mat2

docker run --gpus all --network host -it   -e DISPLAY=$DISPLAY   -e XAUTHORITY=/tmp/.docker.xauth   -v $HOME/.Xauthority:/tmp/.docker.xauth:ro   mat2 bash