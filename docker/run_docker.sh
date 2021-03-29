#!/bin/bash

BASEDIR="$( cd "$(dirname "$0")" ; pwd -P )"

export UID=$UID


#################### change this for your project ###############################
export RAW_PATH=/data_ssd/uia94835/kitti_raw 
export DEPTH_PATH=/data_ssd/uia94835/kitti_depth
export RGB_PATH=/data_ssd/uia94835/kitti_rgb-self-supervised-depth-completion/kitti-rgb 

export RESULTS_PATH=../../results
#################################################################################


NUMBER="$1"

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
    -h|--help)
    usage
    exit
    ;;
esac
done

# (remove) write out exited containers
echo "Exited dockers: "$(docker ps -qa -f status=exited -f name="sr_self_sup")

# read container numbers in use
readarray -t a <<< "$(docker ps -a --format '{{.Names}}' | grep -oP sr_self_sup'.+?\K\d+$')"

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


# Get the latest image name and use it to start the docker container
export DOCKER_IMAGE_NAME=$(docker images | awk '/rose_cube/ {print $1":"$2}' | head -n 1)
echo "The newest docker image is: " $DOCKER_IMAGE_NAME

# run the docker
$BASEDIR/docker-compose -f $BASEDIR/docker-compose.yml run --name rc_sr_self_sup_${NUMBER} rc_sr_self_sup_service