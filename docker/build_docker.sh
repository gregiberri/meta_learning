#!/bin/bash

BASEDIR="$( cd "$(dirname "$0")" ; pwd -P )"

# Get the latest image name and use it to start the docker container
export DOCKER_IMAGE_NAME=depth_prediction
echo "The newest docker image is: " $DOCKER_IMAGE_NAME

export sshd_port=1234

$BASEDIR/docker-compose -f $BASEDIR/docker-compose.yml build depth_prediction_service

