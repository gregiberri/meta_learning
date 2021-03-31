#!/bin/bash

BASEDIR="$( cd "$(dirname "$0")" ; pwd -P )/.."

export UID=$UID
export ORIG_USER_GID=$(id -g $UID)
export ORIG_DEV_GID=$(cat /etc/group | grep developers | cut -d: -f3)


#################### change this for your project ###############################
export IMAGENET_PATH=/raid/data/imagenet/
export IMAGENET84_PATH=/raid/data/imagenet84/

export RESULTS_PATH=/raid/pointcloud/meta_learning/results

#################################################################################



# help function
usage()
{
    printf "Usage:  run_docker.sh \n \
    Runs the docker container and adds the given number to its name (or the next free number). \n \n \
    -n, --number \t numbering in the name of the container \n \
    -h, --help \t help for usage \n"
}

# default port number
sshd_port=1334

# handle arguments
while [ $# -gt 0 ]
do
key="$1"

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
    echo this
    sshd_port="$2"    
    shift # past argument
    shift # past value
    ;;
    
    -h|--help)
    usage
    exit
    ;;
esac
done

# (remove) write out exited containers
echo "Exited dockers: "$(docker ps -qa -f status=exited -f name="meta_learning")

# read container numbers in use
readarray -t a <<< "$(docker ps -a --format '{{.Names}}' | grep -oP meta_learning'.+?\K\d+$')"

# add the number in the argument to the container or the next free number
if [ -z ${NUMBER} ]; then 
  # if number not provided, add the next available
  NUMBER=1
  while true
  do
    isin=false  
    for element in "${a[@]}"
    do
      if [[ $element == $NUMBER ]] ; then
        NUMBER=$(($NUMBER+1))
        isin=true
        break
      fi
    done
    if ! $isin ; then
      break
    fi
  done
  echo "Container number is: "$NUMBER
else
  # if number is provided, try to use it
  isin=false  
  for element in "${a[@]}"
  do
    if [[ $element == $NUMBER ]] ; then
      isin=true
      break
    fi
  done
  if ! $isin ; then
    echo "Container number is: "$NUMBER
  else
    echo "The given container number "$NUMBER" is taken. Choose another one or leave it blank for using the next available number for container number."
    exit
  fi
fi

export NUMBER
export sshd_port

# Get the latest image name and use it to start the docker container
export DOCKER_IMAGE_NAME=$(docker images | awk '/meta_learning/ {print $1":"$2}' | head -n 1)
echo "The newest docker image is: " $DOCKER_IMAGE_NAME

# run the docker
$BASEDIR/docker-compose -f $BASEDIR/docker-compose.yml run --name meta_learning_${NUMBER} meta_learning_service
