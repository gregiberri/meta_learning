#!/bin/bash

BASEDIR="$( cd "$(dirname "$0")" ; pwd -P )"

export UID=$UID
export ORIG_USER_GID=$(id -g $UID)
export ORIG_DEV_GID=$(cat /etc/group | grep developers | cut -d: -f3)


#################### change this for your project ###############################
export RAW_PATH=/raid/data/kitti_raw/
export RGB_PATH=/raid/pointcloud/sr_self_sup/data/self-supervised-depth-completion/kitti-rgb/
export DEPTH_PATH=/raid/data/kitti_depth/
export SEMSEG_PATH=/raid/data/kitti_semseg/

#################################################################################



# help function
usage()
{
    printf "Usage:  run_docker.sh \n \
    Runs the docker container and adds the given number to its name (or the next free number). \n \n \
    -n, --number \t numbering in the name of the container \n \
    -h, --help \t help for usage \n"
}

# handle arguments
while [ $# -gt 0 ]
do
key="$1"

# default port number
sshd_port=1234

case $key in
    -n|--number)
    if ! [[ "$2" =~ ^[0-9]+$ ]]
    then
        echo "Use integer as number"
        exit
    fi
    NUMBER="$2"       
    shift # past argument
    shift # past value
    ;;
    
    -p|--port)
    if ! [[ "$2" =~ ^[0-9]+$ ]]
    then
        echo "Use integer as port"
        exit
    fi
    sshd_port=$2    
    shift # past argument
    shift # past value
    ;;
    
    -h|--help)
    usage
    exit
    ;;
esac
done

export NUMBER
export sshd_port

# Get the latest image name and use it to start the docker container
export DOCKER_IMAGE_NAME=depth_prediction
echo "The newest docker image is: " $DOCKER_IMAGE_NAME

# run the docker
$BASEDIR/docker-compose -f $BASEDIR/docker-compose.yml run --name depth_prediction_${NUMBER} depth_prediction_service
